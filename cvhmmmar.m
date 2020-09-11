function [mcv,cv,rmcv_rand,rcv_rand,rmcv_train,rcv_train] = cvhmmmar(data,T,options)
%
% Obtains the cross-validated sum of prediction quadratic errors.
%
% INPUT
% data          observations, either a struct with X (time series) and C (classes, optional)
%                             or just a matrix containing the time series
% T             length of series
% options       structure with the training options - see documentation.
%               Must contain a field cvfolds, with the number of CV folds,
%               or with a CV folds structure as returned by cvpartition
%
% OUTPUT
% mcv      the averaged cross-validated likelihood and/or fractional mean squared error
% cv       the averaged cross-validated likelihood and/or fractional mean squared error per fold
% rmcv_rand     the mean ratio of mcv for the found solution by the mcv for a random solution
%               log(test / random) 
% rcv_rand      the ratio of mcv for the found solution by the mcv for a random solution
%               log(test / random) 
% mtr       the average training likelihood and/or fractional mean squared error
% rmcv_train   the mean ratio of train / test, which can considered an index of overfitting 
%               log(train / test)
% rcv_train    the ratio of train / test, which can considered an index of overfitting
%               log(train / test)
%
% Author: Diego Vidaurre, OHBA, University of Oxford

if iscell(T)
    if size(T,1)==1, T = T'; end
    for i = 1:length(T)
        if size(T{i},1)==1, T{i} = T{i}'; end
    end
end
N = length(T);

if isstruct(data) && isfield(data,'C')
    warning('C will not be used here')
end

get_ratio_rand = nargout > 2; 
get_ratio_train = nargout > 4; 

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

if ~isobject(options.cvfolds)
    %options.cvfolds = crossvalind('Kfold', length(T), options.cvfolds);
    options.cvfolds = cvpartition(length(T),'KFold',options.cvfolds);
end
nfolds = options.cvfolds.NumTestSets;
orders = formorders(options.order,options.orderoffset,options.timelag,options.exptimelag);
Sind = formindexes(orders,options.S);
if ~options.zeromean, Sind = [true(1,size(Sind,2)); Sind]; end
maxorder = options.maxorder;
cv = zeros(nfolds,1);
rcv_rand = zeros(nfolds,1);
rcv_train = zeros(nfolds,1);

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
        options.A = highdim_pca(data.X,T,options.pca,0,0,0,options.varimax);
    end
else
    options.ndim = size(data.X,2);
end
% Downsampling
if options.downsample > 0
    [data,T] = downsampledata(data,T,options.downsample,options.Fs);
    options.downsample = 0;
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
            
    if get_ratio_rand
        options_r = options;
        options_r.cyc = 1;
        options_r.inittype = 'random';
        options_r.updateGamma = 0;
        if isfield(options_r,'orders')
            options_r = rmfield(options_r,'orders');
        end
        if isfield(options_r,'maxorder')
            options_r = rmfield(options_r,'maxorder');
        end
        hmmtr_r = hmmmar (datatr,Ttr,options_r);
        [~,~,~,LL_r] = hsinference(datate,Tte,hmmtr_r); % LL is the sum of the loglikelihoods 
        LL_r = sum(LL_r) / size(datate.X,1); % get average 
    end
            
    if options.verbose, fprintf('CV fold %d, repetition %d \n',fold); end
    
    if isfield(options,'orders')
        options = rmfield(options,'orders');
    end
    if isfield(options,'maxorder')
        options = rmfield(options,'maxorder');
    end
    hmmtr = hmmmar (datatr,Ttr,options); 
    hmmtr.train.Sind = Sind;
    hmmtr.train.maxorder = maxorder;
    
    % test
    [~,~,~,LL] = hsinference(datate,Tte,hmmtr); % LL is the sum of the loglikelihoods
    cv(fold) = sum(LL) / size(datate.X,1); % get average
    if get_ratio_rand
        rcv_rand(fold) = cv(fold) - LL_r; % log(test / random)
    end
    % train
    if get_ratio_train
        [~,~,~,LL] = hsinference(datatr,Ttr,hmmtr);
        LL = sum(LL) / size(datatr.X,1); % get average
        rcv_train(fold) = LL - cv(fold); % log(train / test)
        %if rcv_train(fold)<0, keyboard; end
    end
    
end

mcv = mean(cv); % mean average LL
rmcv_rand = mean(rcv_rand); % mean ratio
rmcv_train = mean(rcv_train); % mean ratio


end
