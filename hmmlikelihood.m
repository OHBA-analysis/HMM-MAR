function [L,unormGamma,out_precision] = hmmlikelihood(data,T,hmm,preproc)
%
% Get likelihood time series
%
% INPUT
% data          observations, either a struct with X (time series) and C (classes, optional)
%                             or just a matrix containing the time series
% T             length of series
% hmm           hmm data structure
% preproc       whether we should perform the preprocessing options with
%               which the hmm model was trained; 1 by default.
%
% OUTPUT
% L         (T x K) Likelihoods
%
% Author: Diego Vidaurre

% to fix potential compatibility issues with previous versions
hmm = versCompatibilityFix(hmm);

if nargin<4 || isempty(preproc), preproc = 1; end

stochastic_learn = isfield(hmm.train,'BIGNbatch') && hmm.train.BIGNbatch < length(T);
mixture_model = isfield(hmm.train,'id_mixture') && hmm.train.id_mixture;
p = hmm.train.lowrank; do_HMM_pca = (p > 0);

if stochastic_learn, error('hmmlikelihood not yet implemented for stochastic runs'); end

if mixture_model && type==1
    error('Viterbi path not implemented for mixture model')
end

if xor(iscell(data),iscell(T)), error('data and T must be cells, either both or none of them.'); end

if iscell(T)
    for i = 1:length(T)
        if size(T{i},1)==1, T{i} = T{i}'; end
    end
    if size(T,1)==1, T = T'; end
    T = cell2mat(T);
end
checkdatacell;
N = length(T);


if preproc % Adjust the data if necessary
    train = hmm.train;
    checkdatacell;
    data = data2struct(data,T,train);
    % Standardise data and control for ackward trials
    data = standardisedata(data,T,train.standardise);
    % Filtering
    if ~isempty(train.filter)
        data = filterdata(data,T,train.Fs,train.filter);
    end
    % Detrend data
    if train.detrend
        data = detrenddata(data,T);
    end
    % Leakage correction
    if train.leakagecorr ~= 0
        data = leakcorr(data,T,train.leakagecorr);
    end
    % Hilbert envelope
    if train.onpower
        data = rawsignal2power(data,T);
    end
    % Leading Phase Eigenvectors
    if train.leida
        data = leadingPhEigenvector(data,T);
    end
    % pre-embedded  PCA transform
    if length(train.pca_spatial) > 1 || train.pca_spatial > 0
        if isfield(train,'As')
            data.X = bsxfun(@minus,data.X,mean(data.X));
            data.X = data.X * train.As;
        else
            [train.As,data.X] = highdim_pca(data.X,T,train.pca_spatial);
        end
    end
    % Embedding
    if length(train.embeddedlags) > 1
        [data,T] = embeddata(data,T,train.embeddedlags);
    end
    % PCA transform
    if length(train.pca) > 1 || train.pca > 0
        if isfield(train,'A')
            data.X = bsxfun(@minus,data.X,mean(data.X));
            data.X = data.X * train.A;
        else
            [train.A,data.X] = highdim_pca(data.X,T,train.pca,0,0,0,train.varimax);
        end
        % Standardise principal components and control for ackward trials
        data = standardisedata(data,T,train.standardise_pc);
        train.ndim = size(train.A,2);
        train.S = ones(train.ndim);
        orders = formorders(train.order,train.orderoffset,train.timelag,train.exptimelag);
        train.Sind = formindexes(orders,train.S) == 1;
    end
    % Downsampling
    if train.downsample > 0
        [data,T] = downsampledata(data,T,train.downsample,train.Fs);
    end
end

K = length(hmm.state);

if ~isfield(hmm.train,'Sind')
    orders = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
    hmm.train.Sind = formindexes(orders,hmm.train.S) == 1;
end

if ~isfield(hmm,'P')
    hmm = hmmhsinit(hmm);
end

order = hmm.train.maxorder;


if isstruct(data)
    data = data.X;
end
if ~do_HMM_pca
    residuals =  getresiduals(data,T,hmm.train.S,hmm.train.maxorder,hmm.train.order,...
        hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag,hmm.train.zeromean);
end

L = zeros(sum(T)-length(T)*order,K);
tacc = 0;

for n = 1:N
    
    if n==1, t0 = 0; s0 = 0;
    else, t0 = sum(T(1:n-1)); s0 = t0 - order*(n-1);
    end
    
    if do_HMM_pca
        B = obslike(data(t0+1:t0+T(n),:),hmm,[],[],[],1);
    else
        %B = obslike(data(t0+1:t0+T(n),:),hmm,residuals(s0+1:s0+T(n)-order,:));
        if ~isfield(hmm.train,'distribution') || strcmp(hmm.train.distribution,'Gaussian')
            B = obslike(data(t0+1:t0+T(n),:),hmm,residuals(s0+1:s0+T(n)-order,:),[],[],1);
        elseif strcmp(hmm.train.distribution ,'bernoulli')
            B = obslikebernoulli(data(t0+1:t0+T(n),:),hmm,[],[],[],1);
        elseif strcmp(hmm.train.distribution ,'poisson')
            B = obslikepoisson(data(t0+1:t0+T(n),:),hmm,[],[],[],1);
        elseif strcmp(hmm.train.distribution ,'logistic')
            B = obslikelogistic([],hmm,residuals(s0+1:s0+T(n)-order,:),XX,[],1);
        end
    end
    B(B<realmin) = realmin;
    
    L( (1:(T(n)-order)) + tacc ,: ) = B(1+order:end,:);
    
    
    if nargout > 1
        
        P = hmm.P; Pi = hmm.Pi;
        
        scale = zeros(T(n),1);
        alpha = zeros(T(n),K);
        beta = zeros(T(n),K);
        
        alpha(1+order,:) = Pi.*B(1+order,:);
        scale(1+order) = sum(alpha(1+order,:));
        alpha(1+order,:) = alpha(1+order,:)/scale(1+order);
        for i = 2+order:T(n)
            alpha(i,:) = (alpha(i-1,:)*P).*B(i,:);
            scale(i) = sum(alpha(i,:));		% P(X_i | X_1 ... X_{i-1})
            alpha(i,:) = alpha(i,:)/scale(i);
        end
        
        scale(scale<realmin) = realmin;
        
        beta(T(n),:) = ones(1,K)/scale(T(n));
        for i = T(n)-1:-1:1+order
            beta(i,:) = (beta(i+1,:).*B(i+1,:))*(P')/scale(i);
            beta(i,beta(i,:)>realmax) = realmax;
        end
        
        if any(isnan(beta(:)))
            warning(['State time courses became NaN (out of precision). ' ...
                'There are probably extreme events in the data. ' ...
                'Using an approximation..' ])
            unormGamma( (1:(T(n)-order)) + tacc ,: ) = alpha(1+order:end,:); 
            out_precision = true;
        else
            unormGamma( (1:(T(n)-order)) + tacc ,: ) = ...
                alpha(1+order:end,:).*beta(1+order:end,:); 
            out_precision = false;
            
        end
        
    end
    
    tacc = tacc + T(n)-order;
    
end
    
end


