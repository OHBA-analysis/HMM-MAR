function [options,data] = checkoptions (options,data,T,cv)

% Basic checks
if nargin<4, cv = 0; end
if isempty(strfind(which('pca'),matlabroot)) && ~isdeployed
    error(['Function pca() seems to be other than Matlab''s own - you need to rmpath() it. ' ...
        'Use ''rmpath(fileparts(which(''pca'')))'''])
end
if ~isfield(options,'K'), error('K was not specified'); end
if ~isfield(options,'order')
    options.order = 0;
    %if ~isfield(options,'leida') || ~options.leida
    %    warning('order was not specified - it will be set to 0'); 
    %end
end
if options.K<1, error('K must be higher than 0'); end
stochastic_learning = isfield(options,'BIGNbatch') && ...
    (options.BIGNbatch < length(T) && options.BIGNbatch > 0);
if ~stochastic_learning && ~isempty(T)
    if ~isstruct(data), data = struct('X',data); end
    if size(data.X,1)~=sum(T)
        error('Total time specified in T does not match the size of the data')
    end
end

% data options
if ~isfield(options,'Fs'), options.Fs = 1; end
if ~isfield(options,'onpower'), options.onpower = 0; end
if ~isfield(options,'leida'), options.leida = 0; end
if ~isfield(options,'embeddedlags') || isempty(options.embeddedlags) 
    options.embeddedlags = 0; 
end
if ~isfield(options,'embeddedlags_batched') || isempty(options.embeddedlags_batched) 
    options.embeddedlags_batched = 0; 
elseif (-options.embeddedlags_batched(1)) ~= (options.embeddedlags_batched(end))
    error('embeddedlags_batched must by symmetric around 0 and have 1 lag step')
elseif any(diff(options.embeddedlags_batched)~=1)
    error('embeddedlags_batched must by symmetric around 0 and have 1 lag step')
end
if ~isfield(options,'pca'), options.pca = 0; end
if ~isfield(options,'pca_spatial'), options.pca_spatial = 0; end
if ~isfield(options,'lowrank'), options.lowrank = 0; end
if ~isfield(options,'varimax'), options.varimax = 0; end
if ~isfield(options,'maxFOth'), options.maxFOth = Inf; end
if ~isfield(options,'pcamar'), options.pcamar = 0; end
if ~isfield(options,'pcapred'), options.pcapred = 0; end
if ~isfield(options,'vcomp') && options.pcapred>0, options.vcomp = 1; end
if ~isfield(options,'filter'), options.filter = []; end
if ~isfield(options,'detrend'), options.detrend = 0; end
if ~isfield(options,'downsample'), options.downsample = 0; end
if ~isfield(options,'leakagecorr'), options.leakagecorr = 0; end
if ~isfield(options,'sequential'), options.sequential = 0; end

if ~isfield(options,'standardise')
    if ~isfield(options,'distribution') || strcmp(options.distribution,'Gaussian')
        options.standardise = 1;
    else
        options.standardise = 0;
    end
end
if ~isfield(options,'standardise_pc')
    options.standardise_pc = length(options.embeddedlags)>1; 
end
if ~isfield(options,'regularisation'), options.regularisation = 'ARD'; end
if ~isfield(options,'Gamma_constraint'), options.Gamma_constraint = []; end
if ~isfield(options,'Gamma_constraint_radius'), options.Gamma_constraint_radius = []; end
if ~isempty(options.Gamma_constraint_radius)
    if isfield(options,'order') && options.order > 0
        ttrial = T(1) - options.order;
    elseif isfield(options,'embeddedlags') && ~isempty(options.embeddedlags)
        ttrial = T(1) + options.embeddedlags(1) - options.embeddedlags(end); 
    else
        ttrial = T(1);
    end
    options.Gamma_constraint = zeros(ttrial,options.K);
    for k = 1:options.K
        m = (k-0.5)*(ttrial/options.K);
        for t = 1:ttrial
            options.Gamma_constraint(t,k) = exp(-( (t-m).^2) / ...
                options.Gamma_constraint_radius); 
        end
    end
end
if length(options.embeddedlags)>1 && isfield(options,'covtype') && ...
        ~strcmpi(options.covtype,'full')
    options.covtype = 'full';
    warning('options.covtype will be set to ''full'' because embeddedlags is used')
end
if isfield(options,'covtype') && strcmp(options.covtype,'uniquefull')
    options.covtype = 'sharedfull'; 
end
if isfield(options,'covtype') && strcmp(options.covtype,'uniquediag')
    options.covtype = 'shareddiag'; 
end
if ~isfield(options,'acrosstrial_constrained'), options.acrosstrial_constrained = 0; end
if options.acrosstrial_constrained && stochastic_learning
    error('acrosstrial_constrained not implemented (or necessary) for stochastic learning')
end
if ~stochastic_learning && ~isempty(T) && any(T(1)~=T) && options.acrosstrial_constrained
    error('If acrosstrial_constrained, all elements in T must have the same length')
end

% display options
if ~isfield(options,'plotGamma'), options.plotGamma = 0; end

% classic HMM or mixture?
if ~isfield(options,'id_mixture'), options.id_mixture = 0; end
% standard HMM or ehmm
if ~isfield(options,'episodic'), options.episodic = 0; end
% clustering of time series? 
if ~isfield(options,'cluster'), options.cluster = 0; end

if options.episodic % ehmm specific things
    if options.lowrank > 0
        error('Additive probabilistic PCA is not yet inplemented')
    end
    if options.K == 1
        error('A ehmm model must have more than options.K=1 chain')
    end
    if options.sequential
        error('options.episodic and options.sequential are not compatible')
    end
    if isfield(options,'Pstructure') && ~all(options.Pstructure(:)==1)
        error('Chains cannot be constrained with Pstructure in the ehmm model')
    end
    if isfield(options,'Pistructure') && ~all(options.Pistructure(:)==1)
        error('Chains cannot be constrained with Pistructure in the ehmm model')
    end
    if ~isfield(options,'ehmm_priorOFFvsON')
        options.ehmm_priorOFFvsON = 10 * options.K; 
    end % K*10 times more likely to be OFF
    if ~isfield(options,'ehmm_regularisation_baseline')
        options.ehmm_regularisation_baseline = 0.1; 
    end % K*10times more likely to be OFF
end
if options.lowrank > 0 && options.acrosstrial_constrained
    error('HMM_PCA not implemented for acrosstrial_constrained'); 
end

    
% stochastic options
if stochastic_learning
    if options.episodic  
        error('Stochastic inference not yet implemented for ehmm')
    end
    if options.K==1
        error('There is no purpose on using stochastic inference with K=1. Please restart')
    end
    if length(T)==1
        error('It is not possible to run stochastic inference with just one subject - remove option BIGNbatch')
    end
    if options.BIGNbatch < 1
        new_BIGNbatch = min(length(T)/2,10); 
        warning(['BIGNbatch needs to be > 0. Setting to ' num2str(new_BIGNbatch) ])
        options.BIGNbatch = new_BIGNbatch;
    end 
    if ~isfield(options,'BIGNinit') || isempty(options.BIGNinit) 
        if ~isfield(options,'BIGNinitbatch') || isempty(options.BIGNinitbatch) 
            options.BIGNinitbatch = options.BIGNbatch; 
        end
        options.BIGNinit = round(length(T)/options.BIGNinitbatch);
    else
        if isfield(options,'BIGNinitbatch') && ~isempty(options.BIGNinitbatch)
            warning('Both BIGNinit and BIGNinitbatch were specified. Ignoring BIGNinitbatch')
        end
        options.BIGNinitbatch = [];
    end 
    if ~isfield(options,'BIGprior'), options.BIGprior = []; end
    if ~isfield(options,'BIGcyc'), options.BIGcyc = 200; end
    if ~isfield(options,'BIGmincyc'), options.BIGmincyc = 10; end
    if ~isfield(options,'BIGundertol_tostop'), options.BIGundertol_tostop = 5; end
    if ~isfield(options,'BIGcycnobetter_tostop'), options.BIGcycnobetter_tostop = 20; end
    if ~isfield(options,'BIGtol'), options.BIGtol = 1e-5; end
    if ~isfield(options,'BIGinitrep'), options.BIGinitrep = 1; end
    if ~isfield(options,'BIGinitcyc'), options.BIGinitcyc = 50; end
    if ~isfield(options,'BIGforgetrate'), options.BIGforgetrate = 0.9; end
    if ~isfield(options,'BIGdelay'), options.BIGdelay = 1; end
    if ~isfield(options,'BIGbase_weights'), options.BIGbase_weights = 0.95; end % < 1 will promote democracy
    if ~isfield(options,'BIGcomputeGamma'), options.BIGcomputeGamma = 1; end
    if ~isfield(options,'BIGdecodeGamma'), options.BIGdecodeGamma = 1; end
    if ~isfield(options,'BIGinitStrategy')
        if options.lowrank == 0
            options.BIGinitStrategy = 1; % run through all subjects
        else
            options.BIGinitStrategy = 2; % run through a subset
        end
    end
    if ~isfield(options,'BIGverbose')
        if isfield(options,'verbose'), options.BIGverbose = options.verbose; 
        else, options.BIGverbose = 1; 
        end
    end
    if ~isfield(options,'initial_hmm'), options.initial_hmm = []; end
    if length(options.BIGbase_weights)==1
        options.BIGbase_weights = options.BIGbase_weights * ones(1,length(T));
    end
    if ~isfield(options,'Gamma'), options.Gamma = []; end
    if ~isfield(options,'hmm'), options.hmm = []; end
    if options.BIGdelay > 1, warning('BIGdelay is recommended to be 1.'); end
    if options.plotGamma 
        options.plotGamma = 0;
        warning('Plotting is not allowed for stochastic learning.'); 
    end
else
%     if options.plotGamma && any(~(T(1)==T))
%         options.plotGamma = 0;
%         warning('plotGamma is designed to average across trials, but trials have not the same length')
%     end
end

% non-stochastic training options
if stochastic_learning
    % these are for the init, which is done in batches
    
    if isfield(options,'cyc') && ~isempty(options.cyc)
        warning('Option cyc has no effect in stochastic learning, use BIGinitcyc')
    end
    options.cyc = [];
    if ~isfield(options,'initcyc'), options.initcyc = 3; end
    if ~isfield(options,'initrep'), options.initrep = 1; end
    if ~isfield(options,'initcriterion'), options.initcriterion = 'FreeEnergy'; end
    if ~isfield(options,'stopcriterion'), options.stopcriterion = 'FreeEnergy'; end
    if ~isfield(options,'verbose'), options.verbose = 0; end
    if ~isfield(options,'useParallel'), options.useParallel = 1; end
    if isfield(options,'inittype') && strcmp(options.inittype,'window')
        error('Window initialisation not allowed in stochastic learning')
    end
    if ~isfield(options,'repetitions'), options.repetitions = 1; end
else
    if isfield(options,'repetitions') && options.repetitions>1
        warning('options.repetitions does not have an effect for non-stochastic learning')
    end
    options.repetitions = 1;
    if ~isfield(options,'cyc'), options.cyc = 500; end
    if ~isfield(options,'initcyc'), options.initcyc = 10; end
    if ~isfield(options,'initrep'), options.initrep = 5; end
    if ~options.episodic
        if ~isfield(options,'initcriterion'), options.initcriterion = 'FreeEnergy'; end
    end
    if ~isfield(options,'verbose'), options.verbose = 1; end
    % the rest of the stuff will get assigned in the recursive calls
    if ~isfield(options,'meancycstop'), options.meancycstop = 1; end
    if ~isfield(options,'cycstogoafterevent'), options.cycstogoafterevent = 20; end
    if ~isfield(options,'initTestSmallerK'), options.initTestSmallerK = false; end
    if ~isfield(options,'inittype')
        if options.initcyc>0 && options.initrep>0, options.inittype = 'hmmmar';
        else, options.inittype = 'random';
        end
    end
    if ~strcmp(options.inittype,'random') && options.initrep == 0
        options.inittype = 'random';
        warning('Non random init was set, but initrep==0')
    end
    if strcmp(options.inittype,'window') && ...
            ((isfield(options,'order') && options.order == 0) || ...
            (isfield(options,'zeromean') && options.zeromean == 0))
        error('Window initialisation can only be used if options.order > 0 and options.zeromean = 1');
    end
    if strcmp(options.inittype,'window') && ~isfield(options,'init_windowsize')
        error('Parameter init_windowsize must be specified for window-based initialisation');
    end
    if ~isfield(options,'useParallel')
        options.useParallel = (length(T)>1);
    end
    if options.episodic && options.initTestSmallerK
        error('options.episodic and options.initTestSmallerK are not compatible')
    end
    if options.episodic
        if ~isfield(options,'stopcriterion'), options.stopcriterion = 'ChGamma'; end
        if ~strcmpi(options.stopcriterion,'FreeEnergy') && ...
                ~strcmpi(options.stopcriterion,'ChGamma') && ...
                ~strcmpi(options.stopcriterion,'LogLik') 
            error('options.stopcriterion must be ''LogLik'', ''FreeEnergy'' or ''ChGamma'' ')
        end
    else
        if ~isfield(options,'stopcriterion'), options.stopcriterion = 'FreeEnergy'; end
        if ~strcmpi(options.stopcriterion,'FreeEnergy') && ...
                ~strcmpi(options.stopcriterion,'ChGamma')
            error('options.stopcriterion must be ''FreeEnergy'' or ''ChGamma'' ')
        end
    end
    if ~isfield(options,'tol')
        if strcmpi(options.stopcriterion,'ChGamma'), options.tol = 0.1; % percentage
        else, options.tol = 1e-4; 
        end
    end
end

% TUDA specific option 
if ~isfield(options,'behaviour'), options.behaviour = []; end
if ~isempty(options.behaviour), options.tudamonitoring = 1;
elseif ~isfield(options,'tudamonitoring'), options.tudamonitoring = 0;
end
if ~isfield(options,'tuda'), options.tuda = 0; end
if options.tudamonitoring && stochastic_learning
   error('Stochastic learning is not currently compatible with TUDA monitoring options') 
end
if ~isfield(options,'distribution')
    options.distribution = 'Gaussian';
elseif ~strcmp(options.distribution,'Gaussian')
    options.covtype=''; %to avoid later errors
    if ~strcmp(options.distribution,{'logistic','poisson','bernoulli'})
        error('options.distribution takes only values Gaussian, logistic, poisson or bernoulli');
    end
end

% Trans prob mat related options
if ~isfield(options,'grouping') || isempty(options.grouping)  
    options.grouping = ones(length(T),1);
%elseif ~all(options.grouping==1) && stochastic_learning
%    warning('grouping option is not yet implemented for stochastic learning')
%    options.grouping = ones(length(T),1); 
elseif ~all(options.grouping==1)
    warning('Option grouping is not currently supported and will not be used')
    options.grouping = ones(length(T),1);
end  
if size(options.grouping,1)==1,  options.grouping = options.grouping'; end

if options.episodic
    options.Pstructure = []; options.Pistructure = [];
elseif options.sequential ~= 0 
    if isfield(options,'Pstructure'), Pstructure = options.Pstructure;
    else, Pstructure = [];
    end
    if isfield(options,'Pistructure'), Pistructure = options.Pistructure;
    else, Pistructure = [];
    end
    options.Pistructure = zeros(1,options.K);
    options.Pistructure(1:abs(options.sequential)) = 1;
    options.Pistructure = logical(options.Pistructure);
    options.Pstructure = logical(eye(options.K));
    for k = 1:abs(options.sequential)
        options.Pstructure = options.Pstructure + diag(ones(options.K-k,1),k);
    end
    if options.sequential < 0
        for k = 1:abs(options.sequential)
            options.Pstructure = options.Pstructure + diag(ones(options.K-k,1),-k);
        end
    end
    if ~isempty(Pstructure) && any(Pstructure(:)~=options.Pstructure(:))
        warning('Pstructure will be ignored because sequential was specified')
    end
    if ~isempty(Pistructure) && any(Pistructure(:)~=options.Pistructure(:))
        warning('Pistructure will be ignored because sequential was specified')
    end    
elseif ~isfield(options,'Pstructure')
    options.Pstructure = true(options.K);
else
    if any(size(options.Pstructure)~=options.K)
        error('The dimensions of options.Pstructure are incorrect')
    end
    if any(sum(options.Pstructure,1)==0) || any(sum(options.Pstructure,2)==0)
       error('No state can have an entire row or column set to zero in Pstructure') 
    end
    for k = 1:options.K, options.Pstructure(k,k) = 1; end
    options.Pstructure = (options.Pstructure~=0);
end
if ~isfield(options,'Pistructure') && ~options.episodic
    options.Pistructure = true(1,options.K);
elseif ~options.episodic
    if length(options.Pistructure) ~= options.K 
        error('The dimensions of options.Pistructure are incorrect')
    end
    options.Pistructure = (options.Pistructure~=0);
end

% Drop states? 
if ~isfield(options,'dropstates')
    options.dropstates = 0;
    %if any(~options.Pstructure), options.dropstates = 0;
    %else, options.dropstates = ~stochastic_learning; end
else
    if options.dropstates && options.episodic
        warning('episodic and dropstates are incompatible options')
        options.dropstates = 0;
    elseif options.dropstates && any(~options.Pstructure(:))
        warning('If Pstructure  has zeros, dropstates must be zero')
        options.dropstates = 0;
    elseif options.dropstates && stochastic_learning 
        warning('With stochastic learning, dropstates is set to 0')
        options.dropstates = 0;
    elseif options.dropstates && strcmpi(options.stopcriterion,'ChGamma')
        warning('With options.stopcriterion=''ChGamma'', dropstates is set to 0 ')
        options.dropstates = 0;
    end
end

% Check integrity of preproc parameters
if ~isempty(options.filter)
    if length(options.filter)~=2, error('options.filter must contain 2 numbers of being empty'); end
    if (options.filter(1)==0 && isinf(options.filter(2)))
        warning('The specified filter does not do anything - Ignoring.')
        options.filter = [];
    elseif ~isinf(options.filter(2)) && (options.filter(2) < options.Fs/2) && options.order >= 1
        warning(['The lowpass cutoff frequency is lower than the Nyquist frequency - ' ...
            'This is discouraged for a MAR model; better to using downsampling.'])
    end
end
if options.downsample > 0 && isstruct(data) && isfield(data,'C')
    warning('The use of downsampling is currently not compatible with specifying data.C');
    data = rmfield(data,'C');
end
if options.downsample > options.Fs
   warning('Data is going to be upsampled') 
end
if options.leakagecorr ~= 0
    tmp = which('ROInets.closest_orthogonal_matrix');
    if isempty(tmp)
       error('For leakage correction, ROInets must be in path') 
    end
end
if options.leida
   if options.onpower
       error('Options leida and onpower are not compatible')
   end
   if options.order > 0
       error('Option leida and order > 0 are not compatible')
   end   
   if options.pca > 0
       error('Options leida and pca are not compatible')
   end
   if isfield(options,'covtype') && ...
           (strcmp(options.covtype,'full') || strcmp(options.covtype,'diag'))
       error('When using leida, covtype cannot be full or diag')
   end
   options.zeromean = 0; 
   if length(options.embeddedlags) > 1
       error('Option leida and embeddedlags are not compatible')
   end
end

nowarnings = false; 
if iscell(data)
    X = loadfile(data{1},T{1},options); 
    ndim = size(X,2);
elseif length(options.pca)==1 && options.pca == 0
    if isstruct(data)
        ndim = length(options.embeddedlags) * size(data.X,2);
    else
        ndim = length(options.embeddedlags) * size(data,2);
    end
elseif options.pca(1) < 1
    nowarnings = true;
    if isstruct(data) % temporal assignment
        ndim = size(data.X,2);
    else
        ndim = size(data,2);
    end
else
    ndim = options.pca;
end

% state parameters
options = checkStateParametrization(options,ndim,nowarnings);  

if ~stochastic_learning
    data = data2struct(data,T,options);
end

% Some hmm model options unrelated to the observational model
if ~isfield(options,'DirichletDiag')
    if options.order > 0
        if iscell(T), sumT = sum(cell2mat(T));
        else, sumT = sum(T);
        end
        %options.DirichletDiag = sumT/5;
        options.DirichletDiag = 100;
        if options.K > 1 
            warning(['With options.order > 0, you might want to specify options.DirichletDiag ' ...
                'to a larger value if your state time courses are too volatile'])
        end
    else
        options.DirichletDiag = 10;
    end
end
if ~isfield(options,'PriorWeightingP'), options.PriorWeightingP = 1; end
if ~isfield(options,'PriorWeightingPi'), options.PriorWeightingPi = 1; end

% Some more hmm model options unrelated to the observational model
if ~isfield(options,'Gamma'), options.Gamma = []; end
if ~isfield(options,'hmm'), options.hmm = []; end
if ~isfield(options,'fehist'), options.fehist = []; end
if ~isfield(options,'updateObs'), options.updateObs = 1; end
if ~isfield(options,'updateGamma'), options.updateGamma = 1; end
if ~isfield(options,'updateP')
    if options.episodic, options.updateP = 0;
    else, options.updateP = options.updateGamma; 
    end
end
if ~isfield(options,'decodeGamma'), options.decodeGamma = 1; end
if ~isfield(options,'keepS_W'), options.keepS_W = 1; end

% Use MEX?
if isfield(options,'useMEX') && options.useMEX==1 && length(unique(options.grouping))==1
    options.useMEX = verifyMEX();
elseif isfield(options,'useMEX') && options.useMEX==1 && length(unique(options.grouping))>1
    warning('useMEX is not implemented when options.grouping is specified')
    options.useMEX = 0; 
else % ~isfield(options,'useMEX')
    options.useMEX = 0; 
end

if stochastic_learning  
    % the rest will be dealt with in the recursive calls
    return
end

% Further checks
%if options.maxorder+1 >= min(T)
%   error('There is at least one trial that is too short for the specified order') 
%end
if options.K~=size(data.C,2), error('Matrix data.C should have K columns'); end
% if options.K>1 && options.updateGamma == 0 && isempty(options.Gamma)
%     warning('Gamma is unspecified, so updateGamma was set to 1');  options.updateGamma = 1; 
% end
if options.updateGamma == 1 && options.K == 1
    %warning('Since K is one, updateGamma was set to 0');  
    options.updateGamma = 0; options.updateP = 0; 
end
if options.updateGamma == 0 && options.repetitions>1
    error('If Gamma is not going to be updated, repetitions>1 is unnecessary')
end

% Check precomputed state time courses
if ~isempty(options.Gamma)
    if length(options.embeddedlags)>1
        loss_t = -options.embeddedlags(1) + options.embeddedlags(end);
        if (size(options.Gamma,1) ~= (sum(T) - loss_t*length(T) )) || ...
                (size(options.Gamma,2) ~= options.K)
            error('The supplied Gamma has not the right dimensions')
        end        
    else
        if (size(options.Gamma,1) ~= (sum(T) - options.maxorder*length(T))) || ...
                (size(options.Gamma,2) ~= options.K)
            error('The supplied Gamma has not the right dimensions')
        end
    end
end

if (length(T) == 1 && options.initrep==1) && options.useParallel == 1
    warning('Only one trial, no use for parallel computing')
    options.useParallel = 0;
end

% CV options
if cv==1
    if ~isfield(options,'cvfolds'), options.cvfolds = length(T); end
    if ~isfield(options,'cvmode'), options.cvmode = 1; end
    if ~isfield(options,'cvverbose'), options.cvverbose = 0; end
    if ~isobject(options.cvfolds)
        if length(options.cvfolds)>1 && length(options.cvfolds)~=length(T) 
            error('Incorrect assigment of trials to folds');
        end
        if length(options.cvfolds)>1 && ~isempty(options.Gamma) 
            error('Set options.Gamma=[] for cross-validating'); 
        end
        if length(options.cvfolds)==1 && options.cvfolds==0
            error('Set options.cvfolds to a positive integer'); 
        end
    end
end

end


function options = checkStateParametrization(options,ndim,nowarnings)

if isfield(options,'embeddedlags') && length(options.embeddedlags)>1 && options.order>0 
    error('Order needs to be zero for multiple embedded lags')
end
if isfield(options,'AR') && options.AR == 1
    if options.order == 0, error('Option AR cannot be 1 if order==0'); end
    options.S = -1*ones(ndim) + 2*eye(ndim);
end

if isfield(options,'pcamar') && options.pcamar>0 
    if options.order==0, error('Option pcamar>0 must be used with some order>0'); end
    if isfield(options,'S') && any(options.S(:)~=1), error('S must have all elements equal to 1 if pcamar>0'); end 
    if isfield(options,'symmetricprior') && options.symmetricprior==1, error('Priors must be symmetric if pcamar>0'); end
    if isfield(options,'uniqueAR') && options.uniqueAR==1, error('pcamar cannot be >0 if uniqueAR is set to 0'); end
end
if isfield(options,'pcapred') && options.pcapred>0 
    if options.order==0, error('Option pcapred>0 must be used with some order>0'); end
    if isfield(options,'S') && any(options.S(:)~=1), error('S must have all elements equal to 1 if pcapred>0'); end 
    if isfield(options,'symmetricprior') && options.symmetricprior==1
        error('Option symmetricprior makes no sense if pcamar>0'); 
    end
    if isfield(options,'uniqueAR') && options.uniqueAR==1, error('pcapred cannot be >0 if uniqueAR is set to 0'); end
end
if length(options.embeddedlags)==1 && options.pca_spatial>0
   warning('pca_spatial only applies when using embedded lags; use pca instead')
   options.pca_spatial = 0;
end
if options.episodic
    if isfield(options,'covtype') && ~strcmp(options.covtype,'shareddiag')
        error('For ehmm, covtype must be shareddiag (other options not yet implemented)')
    end
    options.covtype = 'shareddiag';
end
if ~isfield(options,'covtype') && options.leida 
    options.covtype = 'shareddiag'; 
elseif ~isfield(options,'covtype') && options.lowrank>0
    options.covtype = 'pca'; 
elseif ~isfield(options,'covtype') && (ndim==1 || (isfield(options,'S') ...
&& ~isempty(options.S) && ~all(options.S==1)))
    options.covtype = 'diag'; 
elseif ~isfield(options,'covtype') && options.order>0, options.covtype = 'diag';
elseif ~isfield(options,'covtype') && ndim>1, options.covtype = 'full'; 
elseif ~isfield(options,'covtype'), options.covtype = 'diag';
elseif isfield(options,'covtype') && ...
        ~strcmp(options.covtype,'full') && ...
        ~strcmp(options.covtype,'sharedfull') && ...
        ~strcmp(options.covtype,'uniquefull') && ...
        ~strcmp(options.covtype,'diag') && ...
        ~strcmp(options.covtype,'shareddiag') && ...
        ~strcmp(options.covtype,'uniquediag') 
     error('Invalid value for option covtype: choose one of "full", "sharedfull", "diag" or "shareddiag"') 
elseif (strcmp(options.covtype,'full') || strcmp(options.covtype,'sharedfull')) && ...
        ndim==1 && length(options.embeddedlags)==1
    warning('Covariance can only be diag or shareddiag if data has only one channel')
    if strcmp(options.covtype,'full'), options.covtype = 'diag';
    else, options.covtype = 'shareddiag';
    end
end
if strcmp(options.covtype,'uniquefull'), options.covtype = 'sharedfull'; end
if strcmp(options.covtype,'uniquediag'), options.covtype = 'shareddiag'; end

if ~isfield(options,'zeromean')
    if length(options.embeddedlags)>1, options.zeromean = 1; 
    elseif options.lowrank>0, options.zeromean = 1; 
    elseif options.order>0, options.zeromean = 1; 
    else, options.zeromean = 0; % i.e. when Gaussian or LEiDA
    end
end
if ~isfield(options,'priorcov_rate'), options.priorcov_rate = []; end
if ~isfield(options,'timelag'), options.timelag = 1; end
if ~isfield(options,'exptimelag'), options.exptimelag = 1; end
if ~isfield(options,'orderoffset'), options.orderoffset = 0; end
if ~isfield(options,'symmetricprior'),  options.symmetricprior = 0; end
if ~isfield(options,'uniqueAR'), options.uniqueAR = 0; end
%if ~isfield(options,'crosstermsonly'), options.crosstermsonly = 0; end

if (options.order>0) && (options.order <= options.orderoffset)
    error('order has to be either zero or higher than orderoffset')
end
if (options.order>0) && (options.timelag<1) && (options.exptimelag<=1)
    error('if order>0 then you should specify either timelag>=1 or exptimelag>=1')
end
if ~isfield(options,'S')
    options.S = ones(ndim);
elseif (ndim~=size(options.S,1)) || (ndim~=size(options.S,2))
    error('Dimensions of S are incorrect; must be a square matrix of size nchannels by nchannels')
elseif any(options.S(:)~=1) && ...
        length(options.pca_spatial) > 1 || (options.pca_spatial > 0 && options.pca_spatial ~= 1)
    warning('S cannot have elements different from 1 if PCA is going to be used')
    options.S = ones(size(options.S));
end

S = (options.S==1); 
if options.zeromean==0 && any(sum(options.S)==0)
    warning('Ignoring mean for channels for which all columns of S are zero')
end
if options.uniqueAR==1 && any(options.S(:)~=1)
    warning('S has no effect if uniqueAR=1')
end
if (strcmp(options.covtype,'full') || strcmp(options.covtype,'sharedfull')) && any(options.S(:)~=1)
    if any(options.S(:)==0)
        error('Global modelling of MAR coefficients not supported with full or sharedfull covariance matrix');
    end
    regressed = sum(S,2)>=1;
    regressors = sum(S,1)>=1;
    tmp = double(regressed)*double(regressors) ~= S; 
    if any(tmp(:))
    	error(['Decoding with full or sharedfull covariance matrix requires S ' ...
            'to be in block design: current design not supported']);
    end % test that regressors and regressed are mutually exclusive and exhaustive
    if any((regressed + regressors') ~= 1) 
    	error(['Decoding with full or sharedfull covariance matrix requires S ' ...
            'to be in block design: current design not supported']);
    end
end

orders = formorders(options.order,options.orderoffset,options.timelag,options.exptimelag);
npar = sum(S(:)) * length(orders);
if npar > 250 && (options.K > 1) && all(options.S(:)==1) && ~nowarnings
   warning(['There is a risk of overparametrisation in the MAR model, with ' num2str(npar) ...
       ' parameters per state'])
   disp('The time-delay embedded model if this is the case (use options.embeddedlags)')
end

if ~isfield(options,'prior') || isempty(options.prior)
    options.prior = [];
elseif ~options.uniqueAR && ndim>1
    error('Fixed priors are only implemented for uniqueAR==1 (or just one channel)')
elseif ~isfield(options.prior,'S') || ~isfield(options.prior,'Mu')
    error('You need to specify S and Mu to set a prior on W')
elseif size(options.prior.S,1)~=(length(orders) + ~options.zeromean) ...
        || size(options.prior.S,2)~=(length(orders) + ~options.zeromean)
    error('The covariance matrix of the supplied prior has not the right dimensions')
elseif cond(options.prior.S) > 1/eps
    error('The covariance matrix of the supplied prior is ill-conditioned')
elseif size(options.prior.Mu,1)~=(length(orders) + ~options.zeromean) || size(options.prior.Mu,2)~=1
    error('The mean of the supplied prior has not the right dimensions')
else
    options.prior.iS = inv(options.prior.S);
    options.prior.iSMu = options.prior.iS * options.prior.Mu;
end
if ~issymmetric(options.S) && options.symmetricprior==1
   error('In order to use a symmetric prior, you need S to be symmetric as well') 
end

if (strcmp(options.covtype,'full') || strcmp(options.covtype,'sharedfull')) && options.uniqueAR
    error('covtype must be diag or shareddiag if uniqueAR==1')
end
if options.uniqueAR && ~options.zeromean
    error('When uniqueAR==1, modelling the mean is not yet supported')
end
if (strcmp(options.covtype,'shareddiag') || strcmp(options.covtype,'sharedfull')) && ...
        options.order == 0 && options.zeromean == 1 && options.lowrank == 0
   error('Unique covariance matrix, order=0 and no mean modelling: there is nothing left to drive the states..') 
end

if ~isfield(options,'Sind')
    if options.pcapred>0
        options.Sind = true(options.pcapred,ndim);
    else
        options.Sind = formindexes(orders,options.S)==1;
    end
    if ~options.zeromean, options.Sind = [true(1,ndim); options.Sind]; end
end

if ~strcmp(options.covtype,'pca') && options.lowrank > 0 
    warning('When states are probabilistic PCA models (option.lowrank>0), covtype has no meaning')
    options.covtype='pca';
end
if options.zeromean==0 && options.lowrank > 0 
    error('Currently, lowrank can only be used for zeromean=1')
end
if options.order > 1 && options.lowrank > 0 
    error('lowrank can only be used for order=0')
end
if options.pca>0 && options.lowrank > 0 
    error('lowrank and pca are not compatible')
end
if options.pcamar > 0 && options.pcapred > 0
    error('Options pcamar and pcapred are not compatible')
end
if ~isfield(options,'orders') || ~isfield(options,'maxorder')
    [options.orders,options.maxorder] = ...
        formorders(options.order,options.orderoffset,options.timelag,options.exptimelag);
end

end


function test = issymmetric(A)

B = A';
test = all(A(:)==B(:)); 
end

function isfine = verifyMEX()
isfine = 1;
try
    [~,~,~]=hidden_state_inference_mx(1,1,1,0);
catch
    isfine = 0;
end
end
