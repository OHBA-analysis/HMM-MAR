function [fe,ll] = hmmfe(data,T,hmm,Gamma,Xi,preproc,grouping)
% Computes the Free Energy of an HMM 
%
% INPUT
% data          observations, either a struct with X (time series) and C (classes, optional)
%                             or just a matrix containing the time series
% T            length of series
% hmm          hmm structure
% Gamma        probability of states conditioned on data (optional)
% Xi           joint probability of past and future states conditioned on data (optional)
%
% OUTPUT
% fe         the variational free energy
% ll         log-likelihood per time point 
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2017)

% to fix potential compatibility issues with previous versions
hmm = versCompatibilityFix(hmm); 
mixture_model = isfield(hmm.train,'id_mixture') && hmm.train.id_mixture;
p = hmm.train.lowrank; do_HMM_pca = (p > 0);

if nargin<4, Gamma = []; end 
if nargin<5, Xi = []; end 
if nargin<6 || isempty(preproc), preproc = 1; end 
if nargin<7 , grouping = ones(length(T),1); end
if size(grouping,1)==1,  grouping = grouping'; end

if isstruct(data), data = data.X; end

options = hmm.train;
hmm.train.grouping = grouping;

stochastic_learn = isfield(options,'BIGNbatch') && ...
    (options.BIGNbatch < length(T) && options.BIGNbatch > 0);

if iscell(T)
    for i = 1:length(T)
        if size(T{i},1)==1, T{i} = T{i}'; end
    end
    if size(T,1)==1, T = T'; end
end

if ~stochastic_learn
    if iscell(T)
        T = cell2mat(T);
    end
    checkdatacell;
    %data = data2struct(data,T,hmm.train);
else
    if ~iscell(data)
        N = length(T);
        dat = cell(N,1); TT = cell(N,1);
        for i=1:N
            t = 1:T(i);
            dat{i} = data(t,:); TT{i} = T(i);
            try data(t,:) = [];
            catch, error('The dimension of data does not correspond to T');
            end
        end
        if ~isempty(data)
            error('The dimension of data does not correspond to T');
        end
        data = dat; T = TT; clear dat TT
    end
end

if preproc && ~stochastic_learn
    % Standardise data and control for ackward trials
    data = standardisedata(data,T,options.standardise);
    % Filtering
    if ~isempty(options.filter)
        data = filterdata(data,T,options.Fs,options.filter);
    end
    % Detrend data
    if options.detrend
        data = detrenddata(data,T);
    end
    % Leakage correction
    if options.leakagecorr ~= 0
        data = leakcorr(data,T,options.leakagecorr);
    end
    % Hilbert envelope
    if options.onpower
        data = rawsignal2power(data,T);
    end
    % Leading Phase Eigenvectors
    if options.leida
        data = leadingPhEigenvector(data,T);
    end
    % pre-embedded PCA transform
    if length(options.pca_spatial) > 1 || (options.pca_spatial > 0 && options.pca_spatial ~= 1)
        if isfield(options,'As')
            data.X = bsxfun(@minus,data.X,mean(data.X));
            data.X = data.X * options.As;
        else
            [options.As,data.X] = highdim_pca(data.X,T,options.pca_spatial);
            options.pca_spatial = size(options.As,2);
        end
    end
    % Embedding
    if length(options.embeddedlags)>1
        [data,T] = embeddata(data,T,options.embeddedlags);
    end
    % PCA transform
    if isfield(options,'A')
        data = bsxfun(@minus,data,mean(data)); % must center
        data = data * options.A;
        % Standardise principal components and control for ackward trials
        data = standardisedata(data,T,options.standardise_pc);
    end
    % Downsampling
    if options.downsample > 0
        [data,T] = downsampledata(data,T,options.downsample,options.Fs);
    end
end

% get residuals
if ~stochastic_learn && ~do_HMM_pca
    if isfield(hmm.state(1),'W')
        ndim = size(hmm.state(1).W.Mu_W,2);
    else
        ndim = size(hmm.state(1).Omega.Gam_rate,2);
    end
    if ~isfield(hmm.train,'Sind')
        orders = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
        hmm.train.Sind = formindexes(orders,hmm.train.S);
    end
    if ~hmm.train.zeromean, hmm.train.Sind = [true(1,ndim); hmm.train.Sind]; end
    residuals =  getresiduals(data,T,hmm.train.Sind,hmm.train.maxorder,hmm.train.order,...
        hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag,hmm.train.zeromean);
else
    residuals = [];
end

% get state time courses
if isempty(Gamma) || isempty(Xi)
   if ~(mixture_model && ~isempty(Gamma)) % we have Gamma and Xi is not needed 
       [Gamma,Xi] = hmmdecode(data,T,hmm,0,residuals,0);
   end
end

if stochastic_learn
    if hmm.train.downsample > 0
        downs_ratio = (hmm.train.downsample/hmm.train.Fs);
    else
        downs_ratio = 1;
    end
    Tmat = downs_ratio*cell2mat(T);
    if length(hmm.train.embeddedlags)>1
        maxorder = 0;
        L = -min(hmm.train.embeddedlags) + max(hmm.train.embeddedlags);
        Tmat = Tmat - L;
    else
        maxorder = hmm.train.maxorder;
    end
    %  P/Pi KL
    fe = sum(evalfreeenergy([],[],[],[],hmm,[],[],[0 0 0 1 0]));
    % Gamma entropy&LL
    fe = fe + sum(evalfreeenergy([],Tmat,Gamma,Xi,hmm,[],[],[1 0 1 0 1]));
    tacc = 0; tacc2 = 0; fell = 0; ll = [];
    for i = 1:1:length(T)
        [X,XX,residuals,Ti] = loadfile(data{i},T{i},options);
        t = (1:(sum(Ti)-length(Ti)*maxorder)) + tacc;
        t2 = (1:(sum(Ti)-length(Ti)*(maxorder+1))) + tacc2;
        tacc = tacc + length(t); tacc2 = tacc2 + length(t2);
        if ~isempty(Xi)
            [f,l] = evalfreeenergy(X,Ti,Gamma(t,:),Xi(t2,:,:),hmm,residuals,XX,[0 1 0 0 0]);
            fell = fell + sum(f); % state KL
            ll = [l; ll];
        else
            [f,l] = evalfreeenergy(X,Ti,Gamma(t,:),[],hmm,residuals,XX,[0 1 0 0 0]);
            fell = fell + sum(f); % state KL
            ll = [l; ll];
        end
    end
    fe = fe + fell;
else
    [fe,ll] = evalfreeenergy(data,T,Gamma,Xi,hmm,residuals);
    fe = sum(fe);
end

end
