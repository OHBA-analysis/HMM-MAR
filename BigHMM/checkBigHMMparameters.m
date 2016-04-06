if ~isfield(options,'K'), error('K was not specified');
else K = options.K; end
% Specific BigHMM options
if ~isfield(options,'BIGNbatch'), BIGNbatch = 10;
else BIGNbatch = options.BIGNbatch; end
if ~isfield(options,'BIGuniqueTrans'), BIGuniqueTrans = 0;
else BIGuniqueTrans = options.BIGuniqueTrans; end
if ~isfield(options,'BIGprior'), options.BIGprior = []; end
if ~isfield(options,'BIGcyc'), BIGcyc = 200;
else BIGcyc = options.BIGcyc; end
if ~isfield(options,'BIGmincyc'), BIGmincyc = 50;
else BIGmincyc = options.BIGmincyc; end
if ~isfield(options,'BIGundertol_tostop'), BIGundertol_tostop = 5;
else BIGundertol_tostop = options.BIGundertol_tostop; end
if ~isfield(options,'BIGtol'), BIGtol = 1e-5;
else BIGtol = options.BIGtol; end
if ~isfield(options,'BIGinitcyc'), BIGinitcyc = 4;
else BIGinitcyc = options.BIGinitcyc; end
if ~isfield(options,'BIGforgetrate'), BIGforgetrate = 0.9;
else BIGforgetrate = options.BIGforgetrate; end
if ~isfield(options,'BIGdelay'), BIGdelay = 1;
else BIGdelay = options.BIGdelay; end
if ~isfield(options,'BIGbase_weights'), BIGbase_weights = 0.95; % < 1 will promote democracy
else BIGbase_weights = options.BIGbase_weights; end
if ~isfield(options,'BIGverbose'), BIGverbose = 1;  
else BIGverbose = options.BIGverbose; end

% HMM-MAR options
if ~isfield(options,'zeromean'), options.zeromean = 0; end
if ~isfield(options,'covtype'), options.covtype = 'full'; end
if ~isfield(options,'order'), options.order = 0; end
if ~isfield(options,'orderoffset'), options.orderoffset = 0; end
if ~isfield(options,'timelag'), options.timelag = 1; end
if ~isfield(options,'exptimelag'), options.exptimelag = 0; end
if ~isfield(options,'cyc'), options.cyc = 50; end
if ~isfield(options,'initcyc'), options.initcyc = 10; end
if ~isfield(options,'initrep'), options.initrep = 3; end
options.dropstates = 0; 
options.verbose = 0; % shut up the individual hmmmar output
if options.order>0
    [options.orders,options.order] = formorders(options.order,options.orderoffset,options.timelag,options.exptimelag);
else
    options.orders = [];
end