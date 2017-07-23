function options = checksoptions (options,T)

% Right now the option S is not implemented for stochastic learning

N = length(T);

% data options
if ~isfield(options,'Fs'), options.Fs = 1; end
if ~isfield(options,'embeddedlags'), options.embeddedlags = 0; end
if ~isfield(options,'pca'), options.pca = 0; end
if ~isfield(options,'pca_spatial'), options.pca_spatial = 0; end
if ~isfield(options,'varimax'), options.varimax = 0; end
if ~isfield(options,'pcamar'), options.pcamar = 0; end
if ~isfield(options,'pcapred'), options.pcapred = 0; end
if ~isfield(options,'vcomp') && options.pcapred>0, options.vcomp = 1; end
if ~isfield(options,'onpower'), options.onpower = 0; end
if ~isfield(options,'filter'), options.filter = []; end
if ~isfield(options,'detrend'), options.detrend = 0; end
if ~isfield(options,'downsample'), options.downsample = 0; end
if ~isfield(options,'standardise'), options.standardise = 1; end %(options.pca>0);
if ~isfield(options,'standardise_pc'), options.standardise_pc = 0; end 
if ~isfield(options,'grouping') || isempty(options.grouping)
    options.grouping = ones(length(T),1); 
elseif ~all(options.grouping==1)
    warning('grouping option is not yet implemented for stochastic learning')
    options.grouping = ones(length(T),1); 
else
    options.grouping = ones(length(T),1); 
end  

if ~isempty(options.filter)
    if length(options.filter)~=2, error('options.filter must contain 2 numbers of being empty'); end
   if (options.filter(1)==0 && isinf(options.filter(2)))
       warning('The specified filter does not do anything - Ignoring.')
       options.filter = [];
   elseif (options.filter(2) < options.Fs/2) && options.order >= 1
       warning(['The lowpass cutoff frequency is lower than the Nyquist frequency - ' ... 
           'This is discouraged for a MAR model'])
   end
end

if length(options.embeddedlags)==1 && options.pca_spatial>0
   warning('pca_spatial only applies when using embedded lags; use pca instead')
   options.pca_spatial = 0;
end

if size(options.grouping,1)==1,  options.grouping = options.grouping'; end

if ~isfield(options,'K'), error('K was not specified'); end
% Specific BigHMM options
if ~isfield(options,'BIGNinitbatch'), options.BIGNinitbatch = 1; end
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
if ~isfield(options,'Gamma'), options.Gamma = []; end
if ~isfield(options,'hmm'), options.hmm = []; end

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
if ~isfield(options,'uniqueAR'), options.uniqueAR = 0; end
%if ~isfield(options,'crosstermsonly'), options.crosstermsonly = 0; end

%if isfield(options,'S') && ~all(options.S(:)==1)
%    error('S(i,j)<1 is not yet implemented for stochastic inference')
%end

options.dropstates = 0; 
options.verbose = 0; % shut up the individual hmmmar output
if options.order>0
    [options.orders,options.order] = formorders(options.order,options.orderoffset,options.timelag,options.exptimelag);
else
    options.orders = [];
end

end
