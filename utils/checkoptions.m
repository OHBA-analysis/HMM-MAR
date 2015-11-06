function [options,data] = checkoptions (options,data,T,cv)

if ~isfield(options,'K'), error('K was not specified'); end
if ~isstruct(data), data = struct('X',data); end
if ~isfield(data,'C'), 
    if options.K>1, data.C = NaN(size(data.X,1),options.K); 
    else data.C = ones(size(data.X,1),1); 
    end
end

options = checkMARparametrization(options,[],size(data.X,2));
options.multipleConf = isfield(options,'state');
if options.multipleConf
    options.maxorder = 0;
    for k = 1:options.K
        if ~isempty(options.state(k).train)
            options.state(k).train = checkMARparametrization(options.state(k).train,options.S,size(data.X,2));
            train =  options.state(k).train;
            [~,order] = formorders(train.order,train.orderoffset,train.timelag,train.exptimelag);
            options.maxorder = max(options.maxorder,order);
        end
    end
else
    [~,options.maxorder] = formorders(options.order,options.orderoffset,options.timelag,options.exptimelag);
end

options.ndim = size(data.X,2);
if ~isfield(options,'Fs'), options.Fs = 1; end
if ~isfield(options,'cyc'), options.cyc = 1000; end
if ~isfield(options,'tol'), options.tol = 1e-5; end
if ~isfield(options,'meancycstop'), options.meancycstop = 1; end
if ~isfield(options,'cycstogoafterevent'), options.cycstogoafterevent = 20; end
if ~isfield(options,'initcyc'), options.initcyc = 100; end
if ~isfield(options,'initrep'), options.initrep = 5; end
if ~isfield(options,'inittype'), options.inittype = 'EM'; end
if ~isfield(options,'Gamma'), options.Gamma = []; end
if ~isfield(options,'hmm'), options.hmm = []; end
if ~isfield(options,'fehist'), options.fehist = []; end
if ~isfield(options,'DirichletDiag'), options.DirichletDiag = 2; end
%if ~isfield(options,'whitening'), options.whitening = 0; end
if ~isfield(options,'repetitions'), options.repetitions = 1; end
if ~isfield(options,'updateGamma'), options.updateGamma = 1; end
if ~isfield(options,'keepS_W'), options.keepS_W = 1; end
if ~isfield(options,'verbose'), options.verbose = 1; end

if isempty(options.Gamma) && ~isempty(options.hmm)
    error('Gamma must be provided in options if you want a warm restart')
end

if ~strcmp(options.inittype,'random') && options.initrep == 0,
    options.inittype = 'random';
    warning('Non random init was set, but initrep==0')
end

if options.K~=size(data.C,2), error('Matrix data.C should have K columns'); end
if options.K>1 && options.updateGamma == 0 && isempty(options.Gamma), 
    warning('Gamma is unspecified, so updateGamma was set to 1');  options.updateGamma = 1; 
end
if options.updateGamma == 1 && options.K == 1,
    warning('Since K is one, updateGamma was set to 0');  options.updateGamma = 0; 
end
if options.updateGamma == 0 && options.repetitions>1,
    error('If Gamma is not going to be updated, repetitions>1 is unnecessary')
end

if ~isempty(options.Gamma)
    if (size(options.Gamma,1) ~= (sum(T) - options.maxorder*length(T))) || (size(options.Gamma,2) ~= options.K),
        error('The supplied Gamma has not the right dimensions')
    end
end

if cv==1
    if ~isfield(options,'cvfolds'), options.cvfolds = length(T); end
    if ~isfield(options,'cvrep'), options.cvrep = 1; end
    if ~isfield(options,'cvmode'), options.cvmode = 1; end
    if ~isfield(options,'cvverbose'), options.cvverbose = 0; end
    if length(options.cvfolds)>1 && length(options.cvfolds)~=length(T), error('Incorrect assigment of trials to folds'); end
    if length(options.cvfolds)>1 && ~isempty(options.Gamma), error('Set options.Gamma=[] for cross-validating'); end
    if length(options.cvfolds)==1 && options.cvfolds==0, error('Set options.cvfolds to a positive integer'); end
    if options.K==1 && isfield(options,'cvrep')>1, warning('If K==1, cvrep>1 has no point; cvrep is set to 1 \n'); end
end

end


function options = checkMARparametrization(options,S,ndim)
if ~isfield(options,'order'), error('order was not specified'); end
if ~isfield(options,'covtype') && ndim==1, options.covtype = 'diag'; 
elseif ~isfield(options,'covtype') && ndim>1, options.covtype = 'full'; 
elseif (strcmp(options.covtype,'full') || strcmp(options.covtype,'uniquefull')) && ndim==1
    warning('Covariance can only be diag or uniquediag if data has only one channel')
    if strcmp(options.covtype,'full'), options.covtype = 'diag';
    else options.covtype = 'uniquediag';
    end
end
if ~isfield(options,'zeromean'), 
    if options.order>0, options.zeromean = 1; 
    else options.zeromean = 0;
    end
end
if ~isfield(options,'embeddedlags'), options.embeddedlags = 0; end
if ~isfield(options,'timelag'), options.timelag = 1; end
if ~isfield(options,'exptimelag'), options.exptimelag = 1; end
if ~isfield(options,'orderoffset'), options.orderoffset = 0; end
if ~isfield(options,'symmetricprior'), options.symmetricprior = 1; end
if ~isfield(options,'uniqueAR'), options.uniqueAR = 0; end
if (options.order>0) && (options.order <= options.orderoffset)
    error('order has to be either zero or higher than orderoffset')
end
if (options.order>0 && options.timelag<1 && options.exptimelag<=1)
    error('if order>0 then you should specify either timelag>=1 or exptimelag>=1')
end
if ~isfield(options,'S'), 
    if nargin>=2 && ~isempty(S)
        options.S = S;
    else
        options.S = ones(length(options.embeddedlags)*ndim,length(options.embeddedlags)*ndim); 
    end
elseif nargin>=2 && ~isempty(S) && any(S(:)~=options.S(:))
    error('S has to be equal across states')
elseif options.uniqueAR==1 && any(S(:)~=1)
    warning('S has no effect if uniqueAR=1')
end
orders = formorders(options.order,options.orderoffset,options.timelag,options.exptimelag);
if ~isfield(options,'prior')
    options.prior = [];
elseif ~options.uniqueAR && ndim>1
    error('Fixed priors are only implemented for uniqueAR==1 (or just one channel)')
elseif ~isfield(options.prior,'S') || ~isfield(options.prior,'Mu')
    error('You need to specify S and Mu to set a prior on W')
elseif size(options.prior.S,1)~=(length(orders) + ~options.zeromean) ...
        || size(options.prior.S,2)~=(length(orders) + ~options.zeromean)
    error('The covariance matrix of the supplied prior has not the right dimensions')
elseif cond(options.prior.S) > 1/eps;
    error('The covariance matrix of the supplied prior is ill-conditioned')
elseif size(options.prior.Mu,1)~=(length(orders) + ~options.zeromean) || size(options.prior.Mu,2)~=1
    error('The mean of the supplied prior has not the right dimensions')
else
    options.prior.iS = inv(options.prior.S);
    options.prior.iSMu = options.prior.iS * options.prior.Mu;
end
if ~issymmetric(options.S) && options.symmetricprior==0,
   error('In order to use a symmetric prior, you need S to be symmetric as well') 
end
if (strcmp(options.covtype,'full') || strcmp(options.covtype,'uniquefull')) &&  ~all(options.S(:)==1),
    error('if S is not all set to 1, then covtype must be diag or uniquediag')
end
if (strcmp(options.covtype,'full') || strcmp(options.covtype,'uniquefull')) && options.uniqueAR,
    error('covtype must be diag or uniquediag if uniqueAR==1')
end
if options.uniqueAR && ~options.zeromean
    error('When unique==1, modelling the mean is not yet supported')
end
options.Sind = formindexes(orders,options.S);
if ~options.zeromean, options.Sind = [true(1,ndim); options.Sind]; end
end


function test = issymmetric(A)
B = A';
test = all(A(:)==B(:)); 
end

