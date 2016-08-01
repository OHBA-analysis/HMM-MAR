function options = checksoptions (options,T)

% Right now the option S is not implemented for stochastic learning

N = length(T);

% data options
if ~isfield(options,'embeddedlags'), options.embeddedlags = 0; end
if ~isfield(options,'pca'), options.pca = 0; end
if ~isfield(options,'pcamar'), options.pcamar = 0; end
if ~isfield(options,'standardise'), options.standardise = (options.pca>0); end

if ~isfield(options,'K'), error('K was not specified'); end
% Specific BigHMM options
if ~isfield(options,'BIGNbatch'), options.BIGNbatch = min(10,N); end
if ~isfield(options,'BIGuniqueTrans'), options.BIGuniqueTrans = 1; end
if ~isfield(options,'BIGprior'), options.BIGprior = []; end
if ~isfield(options,'BIGcyc'), options.BIGcyc = 200; end
if ~isfield(options,'BIGmincyc'), options.BIGmincyc = 10; end
if ~isfield(options,'BIGundertol_tostop'), options.BIGundertol_tostop = 5; end
if ~isfield(options,'BIGcycnobetter_tostop'), options.BIGcycnobetter_tostop = 20; end
if ~isfield(options,'BIGtol'), options.BIGtol = 1e-5; end
if ~isfield(options,'BIGinitrep'), options.BIGinitrep = 1; end
if ~isfield(options,'BIGforgetrate'), options.BIGforgetrate = 0.9; end
if ~isfield(options,'BIGdelay'), options.BIGdelay = 1; end
if ~isfield(options,'BIGbase_weights'), options.BIGbase_weights = 0.95; end % < 1 will promote democracy
if ~isfield(options,'BIGcomputeGamma'), options.BIGcomputeGamma = 1; end
if ~isfield(options,'BIGdecodeGamma'), options.BIGdecodeGamma = 1; end
if ~isfield(options,'BIGverbose'), options.BIGverbose = 1; end  
if ~isfield(options,'initial_hmm'), options.initial_hmm = []; end
options.BIGbase_weights = options.BIGbase_weights * ones(1,N);

% HMM-MAR options
if ~isfield(options,'zeromean'), options.zeromean = 0; end
if ~isfield(options,'covtype'), options.covtype = 'full'; end
if ~isfield(options,'order'), options.order = 0; end
if ~isfield(options,'orderoffset'), options.orderoffset = 0; end
if ~isfield(options,'timelag'), options.timelag = 1; end
if ~isfield(options,'exptimelag'), options.exptimelag = 0; end
if ~isfield(options,'cyc'), options.cyc = 15; end
if ~isfield(options,'initcyc'), options.initcyc = 5; end
if ~isfield(options,'initrep'), options.initrep = 3; end
if ~isfield(options,'useParallel'), options.useParallel = 1; end
options.dropstates = 0; 
options.verbose = 0; % shut up the individual hmmmar output
if options.order>0
    [options.orders,options.order] = formorders(options.order,options.orderoffset,options.timelag,options.exptimelag);
else
    options.orders = [];
end