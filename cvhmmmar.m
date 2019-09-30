function [mcv,cv] = cvhmmmar(data,T,options)
%
% Obtains the cross-validated sum of prediction quadratic errors.
%
% INPUT
% data          observations, either a struct with X (time series) and C (classes, optional)
%                             or just a matrix containing the time series
% T             length of series
% options       structure with the training options - see documentation
%
% OUTPUT
% mcv      the averaged cross-validated likelihood and/or fractional mean squared error
% cv       the averaged cross-validated likelihood and/or fractional mean squared error per fold
%
% Author: Diego Vidaurre, OHBA, University of Oxford

if iscell(T)
    if size(T,1)==1, T = T'; end
    for i = 1:length(T)
        if size(T{i},1)==1, T{i} = T{i}'; end
        T{i} = int64(T{i});
    end
else
    T = int64(T);
end
N = length(T);

if isstruct(data) && isfield(data,'C')
    error('C cannot be specified within data here')
end

% is this going to be using the stochastic learning scheme? 
stochastic_learn = isfield(options,'BIGNbatch') && (options.BIGNbatch < N && options.BIGNbatch > 0);
if stochastic_learn
    error('Stochastic learning cannot currently be used within CV')
end
options = checkspelling(options);

if xor(iscell(data),iscell(T)), error('X and T must be cells, either both or none of them.'); end

if iscell(T)
    T = cell2mat(T);
end
checkdatacell;
[options,data] = checkoptions(options,data,T,1);

if options.cvmode~=1
    error('The use of options.cvmode different from 1 has been discontinued.')
end

if ~all(options.grouping==1)
    error('grouping option is not yet implemented in cvhmmmar')
end
    
options.verbose = options.cvverbose;
options.dropstates = 0;
options.updateGamma = options.K>1;
options.updateP = options.updateGamma;

mcv = 0; 
if length(options.cvfolds)==1
    %options.cvfolds = crossvalind('Kfold', length(T), options.cvfolds);
    options.cvfolds = cvpartition(length(T),'KFold',options.cvfolds);
end
nfolds = options.cvfolds.NumTestSets;
[orders,order] = formorders(options.order,options.orderoffset,options.timelag,options.exptimelag);
Sind = formindexes(orders,options.S);
if ~options.zeromean, Sind = [true(1,size(Sind,2)); Sind]; end
maxorder = options.maxorder;
cv = zeros(nfolds,1);

%%% Preprocessing
% Standardise data and control for ackward trials
data = standardisedata(data,T,options.standardise);
% Filtering
if ~isempty(options.filter)
    data = filterdata(data,T,options.Fs,options.filter); options.filter = [];
end
% Detrend data
if options.detrend
    data = detrenddata(data,T); options.detrend = 0;
end
% Leakage correction
if options.leakagecorr ~= 0
    data = leakcorr(data,T,options.leakagecorr); options.leakagecorr = 0;
end
% Hilbert envelope
if options.onpower
    data = rawsignal2power(data,T); options.onpower = 0;
end
% Leading Phase Eigenvectors
if options.leida
    data = leadingPhEigenvector(data,T); options.leida = 0;
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
    options.pca_spatial = [];
end
% Embedding
if length(options.embeddedlags) > 1
    [data,T] = embeddata(data,T,options.embeddedlags); options.embeddedlags = 0;
end
% PCA transform
if length(options.pca) > 1 || (options.pca > 0 && options.pca ~= 1)
    if isfield(options,'A')
        data.X = bsxfun(@minus,data.X,mean(data.X));
        data.X = data.X * options.A;
    else
        [options.A,data.X] = highdim_pca(data.X,T,options.pca,0,0,0,options.varimax);
        options.pca = size(options.A,2);
    end
    % Standardise principal components and control for ackward trials
    data = standardisedata(data,T,options.standardise_pc);
    options.ndim = size(options.A,2);
    options.S = ones(options.ndim);
    options.Sind = formindexes(options.orders,options.S);
    if ~options.zeromean, options.Sind = [true(1,size(options.Sind,2)); options.Sind]; end
    options.pca = 0;
else
    options.ndim = size(data.X,2);
end
% Rank reduction
if options.rank > 0
    [options.A,data.X] = highdim_pca(data.X,T,options.rank,0,0,0);
    data.X =  data.X * options.A';
    data.X =  data.X + 1e-2 * randn(size(data.X)); % add some noise to avoid ill-conditioning
    options.rank = 0;
end
% Downsampling
if options.downsample > 0
    [data,T] = downsampledata(data,T,options.downsample,options.Fs);
    options.downsample = 0;
end
% get global eigendecomposition
if options.firsteigv
    if isstruct(data)
        data.X = bsxfun(@minus,data.X,mean(data.X));
        options.gram = data.X' * data.X;
    else
        data = bsxfun(@minus,data,mean(data));
        options.gram = X' * X;
    end
    [options.eigvec,options.eigval] = svd(options.gram);
    options.eigval = diag(options.eigval);
    options.firsteigv = 0;
end
if options.pcamar > 0 && ~isfield(options,'B')
    % PCA on the predictors of the MAR regression, per lag: X_t = \sum_i X_t-i * B_i * W_i + e
    options.B = pcamar_decomp(data,T,options);
    options.pcamar = 0;
end
if options.pcapred > 0 && ~isfield(options,'V')
    % PCA on the predictors of the MAR regression, together:
    % Y = X * V * W + e, where X contains all the lagged predictors
    % So, unlike B, V draws from the temporal dimension and not only spatial
    options.V = pcapred_decomp(data,T,options);
    options.pcapre = 0;
end

for fold = 1:nfolds
    
    indtr = []; Ttr = [];
    indte = []; Tte = []; 
    test = [];
    cvtest = options.cvfolds.test(fold);
    % build fold
    for i = 1:length(T)
        t0 = sum(T(1:(i-1)))+1; t1 = sum(T(1:i)); 
        Ti = t1-t0+1;
        if cvtest(i) % in testing
            indte = [indte (t0:t1)];
            Tte = [Tte Ti];
            test = [test; ones(Ti,1)];
        else % in training
            indtr = [indtr (t0:t1)];
            Ttr = [Ttr Ti];
        end
    end
    datatr.X = data.X(indtr,:); 
    datate.X = data.X(indte,:); 
    %datatr.C = data.C(indtr,:); 
    %datate.C = data.C(indte,:);     
        
    Fe = Inf;
      
    for it = 1:options.cvrep
        
        if options.verbose, fprintf('CV fold %d, repetition %d \n',fold,it); end

        if isfield(options,'orders')
            options = rmfield(options,'orders');
        end
        if isfield(options,'maxorder')
            options = rmfield(options,'maxorder');
        end
        [hmmtr,~,~,~,~,~,fe] = hmmmar (datatr,Ttr,options); fe = fe(end);
        hmmtr.train.Sind = Sind;
        hmmtr.train.maxorder = maxorder;
               
        % test
        if fe < Fe
            Fe = fe;
            [~,~,~,LL] = hsinference(datate,Tte,hmmtr);
            cv(fold) = sum(LL);
        end
        
    end
    
    mcv = mcv + cv(fold);
    
end

end
