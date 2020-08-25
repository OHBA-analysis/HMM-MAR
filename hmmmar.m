function [hmm, Gamma, Xi, vpath, GammaInit, residuals, fehist, feterms, rho] = ...
    hmmmar (data,T,options)
% Main function to train the HMM-MAR model, compute the Viterbi path and,
% if requested, obtain the cross-validated sum of prediction quadratic errors.
%
% INPUT
% data          observations; either a struct with X (time series) and C (classes, optional),
%                             or a matrix containing the time series,
%                             or a list of file names
% T             length of series
% options       structure with the training options - see documentation in 
%                       https://github.com/OHBA-analysis/HMM-MAR/wiki
%
% OUTPUT
% hmm           estimated HMMMAR model
% Gamma         Time courses of the states probabilities given data
% Xi            joint probability of past and future states conditioned on data
% vpath         most likely state path of hard assignments
% GammaInit     Time courses used after initialisation.
% residuals     if the model is trained on the residuals, the value of those
% fehist        historic of the free energies across iterations
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2015)

if iscell(T)
    if size(T,1)==1, T = T'; end
    for i = 1:length(T)
        if size(T{i},1)==1, T{i} = T{i}'; end
    end
end
N = length(T);

% is this going to be using the stochastic learning scheme? 
stochastic_learn = isfield(options,'BIGNbatch') && (options.BIGNbatch < N && options.BIGNbatch > 0);
options = checkspelling(options);
if ~stochastic_learn && ...
        (isfield(options,'BIGNinitbatch') || ...
        isfield(options,'BIGprior') || ...
        isfield(options,'BIGcyc') || ...
        isfield(options,'BIGmincyc') || ...
        isfield(options,'BIGundertol_tostop') || ...
        isfield(options,'BIGcycnobetter_tostop') || ...
        isfield(options,'BIGtol') || ...
        isfield(options,'BIGinitrep') || ...
        isfield(options,'BIGforgetrate') || ...
        isfield(options,'BIGdelay') || ...
        isfield(options,'BIGbase_weights') || ...
        isfield(options,'BIGcomputeGamma') || ...
        isfield(options,'BIGdecodeGamma') || ...
        isfield(options,'BIGverbose'))
    warning(['In order to use stochastic learning, BIGNbatch needs to be specified ' ... 
        '- running standard inference'])
    if isfield(options,'BIGNbatch'), options = rmfield(options,'BIGNbatch'); end
end

% do some data checking and preparation
if xor(iscell(data),iscell(T)), error('X and T must be cells, either both or none of them.'); end
if stochastic_learn % data is a cell, either with strings or with matrices
    if isstruct(data) 
        if isfield(data,'C')
            warning(['The use of semisupervised learning is not implemented for stochatic inference; ' ...
                'removing data.C'])
        end
        data = data.X;
    end
    if ~iscell(data) % make it cell
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
else % data can be a cell or a matrix
    if iscell(T)
        T = cell2mat(T);
    end
    checkdatacell;
end
[options,data] = checkoptions(options,data,T,0);
do_HMM_pca = (options.lowrank > 0);

ver = version('-release');
oldMatlab = ~isempty(strfind(ver,'2010')) || ~isempty(strfind(ver,'2010')) ...
    || ~isempty(strfind(ver,'2011')) || ~isempty(strfind(ver,'2012'));

% set the matlab parallel computing environment
if options.useParallel==1 && usejava('jvm')
    try
        if oldMatlab
            if matlabpool('size')==0
                matlabpool
            end
        else
            gcp;
        end
    catch
        error('Issue with the matlab parallel computing environment - use options.useParallel==0');
    end
end

gatherStats = 0;
if isfield(options,'DirStats')
    profile on
    gatherStats = 1; 
    DirStats = options.DirStats;
    options = rmfield(options,'DirStats'); 
    % to avoid recurrent calls to hmmmar to do the same
end

if stochastic_learn
    
    % get PCA pre-embedded loadings
    if length(options.pca_spatial) > 1 || (options.pca_spatial > 0 && options.pca_spatial ~= 1)
        if ~isfield(options,'As')
            options.As = highdim_pca(data,T,options.pca_spatial,...
                0,options.standardise,...
                options.onpower,0,options.detrend,...
                options.filter,options.leakagecorr,options.Fs);
        end
        options.pca_spatial = size(options.As,2);
    else
        options.As = [];
    end    
    if length(options.embeddedlags) > 1
        elmsg = '(embedded)';
    else
        elmsg = '';
    end
    % get PCA loadings
    if length(options.pca) > 1 || (options.pca > 0 && options.pca ~= 1) || ...
            isfield(options,'A')
        if ~isfield(options,'A')
            [options.A,~,e] = highdim_pca(data,T,options.pca,...
                options.embeddedlags,options.standardise,...
                options.onpower,options.varimax,options.detrend,...
                options.filter,options.leakagecorr,options.Fs,options.As);
            options.pca = size(options.A,2);
            if options.verbose
                if options.varimax
                    fprintf('Working in PCA/Varimax %s space, with %d components. \n',elmsg,options.pca)
                    fprintf('(explained variance = %1f)  \n',e(options.pca))
                else
                    fprintf('Working in PCA %s space, with %d components. \n',elmsg,options.pca)
                    fprintf('(explained variance = %1f)  \n',e(options.pca))
                end
            end
        end
        options.ndim = size(options.A,2);
        options.S = ones(options.ndim);
        options.Sind = formindexes(options.orders,options.S);
        if ~options.zeromean, options.Sind = [true(1,size(options.Sind,2)); options.Sind]; end
    else
        options.As = [];
    end
    if isfield(options,'A') && ~isempty(options.A)
        options.ndim = size(options.A,2);
    elseif isfield(options,'As') && ~isempty(options.As)
        options.ndim = size(options.As,2);
    else
        X = loadfile(data{1},T{1},options); 
        options.ndim = size(X,2);
    end
    if options.pcamar > 0 && ~isfield(options,'B')
        % PCA on the predictors of the MAR regression, per lag: X_t = \sum_i X_t-i * B_i * W_i + e
        options.B = pcamar_decomp(data,T,options);
    end
    if options.pcapred > 0 && ~isfield(options,'V')
        % PCA on the predictors of the MAR regression, together: 
        % Y = X * V * W + e, where X contains all the lagged predictors
        % So, unlike B, V draws from the temporal dimension and not only spatial
        options.V = pcapred_decomp(data,T,options);
    end
     
    if isempty(options.Gamma) && isempty(options.hmm) % both unspecified
        [hmm,info] = hmmsinit(data,T,options);
        GammaInit = []; 
    elseif isempty(options.Gamma) && ~isempty(options.hmm) % Gamma unspecified
        hmm = versCompatibilityFix(options.hmm);
        GammaInit = [];
        [hmm,info] = hmmsinith(data,T,options,hmm);
    else % Gamma specified
        if ~isempty(options.hmm)
           warning('options.hmm will not be used because options.Gamma was specified') 
        end
        GammaInit = options.Gamma;
        options = rmfield(options,'Gamma');
        [hmm,info] = hmmsinitg(data,T,options,GammaInit);
    end % If both are specified, hmm is not used
    if options.BIGcyc>1
        [hmm,fehist,feterms,rho] = hmmstrain(data,T,hmm,info,options);
    else
        fehist = []; feterms = []; rho = [];
    end
    Gamma = []; Xi = []; vpath = []; residuals = [];
    if options.BIGcomputeGamma && nargout >= 2
       Gamma = hmmdecode(data,T,hmm,0); 
       if nargout > 2 
           warning(['When stochastic inference is run, Xi will be returned ' ...
               'as empty to prevent excessive memory usage. ' ...
               'If required, it can be obtained by calling to hmmdecode directly'])
       end
    end
    if options.BIGdecodeGamma && nargout >= 4
       vpath = hmmdecode(data,T,hmm,1); 
    end
    
else
    
    % Standardise data and control for ackward trials
    valid_dims = computeValidDimensions(data,options);
    data = standardisedata(data,T,options.standardise,valid_dims); 
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
    if length(options.embeddedlags) > 1  
        [data,T] = embeddata(data,T,options.embeddedlags);
        elmsg = '(embedded)';
    else
        elmsg = ''; 
    end
    % PCA transform
    if length(options.pca) > 1 || (options.pca > 0 && options.pca ~= 1) || ...
            isfield(options,'A')
        if isfield(options,'A')
            data.X = bsxfun(@minus,data.X,mean(data.X));   
            data.X = data.X * options.A; 
        else
            [options.A,data.X,e] = highdim_pca(data.X,T,options.pca,0,0,0,options.varimax);
            options.pca = size(options.A,2);
            if options.verbose
                if options.varimax
                    fprintf('Working in PCA/Varimax %s space, with %d components. \n',elmsg,options.pca)
                    fprintf('(explained variance = %1f)  \n',e(options.pca))
                else
                    fprintf('Working in PCA %s space, with %d components. \n',elmsg,options.pca)
                    fprintf('(explained variance = %1f)  \n',e(options.pca))
                end
            end
        end
        % Standardise principal components and control for ackward trials
        data = standardisedata(data,T,options.standardise_pc);
        options.ndim = size(options.A,2);
        options.S = ones(options.ndim);
        options.Sind = formindexes(options.orders,options.S);
        if ~options.zeromean, options.Sind = [true(1,size(options.Sind,2)); options.Sind]; end
    else
        options.ndim = size(data.X,2);
    end
    % Downsampling
    if options.downsample > 0 
       [data,T] = downsampledata(data,T,options.downsample,options.Fs); 
    end
    if options.pcamar > 0 && ~isfield(options,'B')
        % PCA on the predictors of the MAR regression, per lag: X_t = \sum_i X_t-i * B_i * W_i + e
        options.B = pcamar_decomp(data,T,options);
    end
    if options.pcapred > 0 && ~isfield(options,'V')
        % PCA on the predictors of the MAR regression, together: 
        % Y = X * V * W + e, where X contains all the lagged predictors
        % So, unlike B, V draws from the temporal dimension and not only spatial
        options.V = pcapred_decomp(data,T,options);
    end    
    
    if isfield(options,'fehist'), fehistInit = options.fehist;
    else, fehistInit = [];
    end
    if isempty(options.Gamma) && isempty(options.hmm) % both unspecified
        if options.K > 1
            Sind = options.Sind;
            if options.initrep>0 && options.initcyc>0 && ...
                    (strcmpi(options.inittype,'HMM-MAR') || strcmpi(options.inittype,'HMMMAR'))
                [GammaInit,fehistInit] = hmmmar_init(data,T,options,Sind);
            elseif strcmpi(options.inittype,'sequential')
                GammaInit = initGamma_seq(T-options.maxorder,options.K);
            elseif options.initrep>0 &&  strcmpi(options.inittype,'EM')
                error('EM init is deprecated; use HMM-MAR initialisation instead')
                %options.nu = sum(T)/200;
                %options.Gamma = em_init(data,T,options,Sind);
            elseif options.initrep>0 && strcmpi(options.inittype,'GMM')
                error('GMM init is deprecated; use HMM-MAR initialisation instead')
                %options.Gamma = gmm_init(data,T,options);
            elseif strcmpi(options.inittype,'random') || options.initrep==0 || options.initcyc==0
                GammaInit = initGamma_random(T-options.maxorder,options.K,...
                    options.DirichletDiag,options.Pstructure,options.Pistructure);
            else
                error('Unknown init method')
            end
        else
            options.Gamma = ones(sum(T)-length(T)*options.maxorder,1);
            GammaInit = options.Gamma;
        end
    elseif isempty(options.Gamma) && ~isempty(options.hmm) % Gamma unspecified, hmm specified
        GammaInit = [];
    else % Gamma specified
        if ~isempty(options.hmm)
           warning('options.hmm will not be used because options.Gamma was specified') 
           options.hmm = [];
        end
        % hmm unspecified, or both specified
        GammaInit = options.Gamma;
    end % If both are specified, hmm is not used
    options = rmfield(options,'Gamma');

    % If initialization Gamma has fewer states than options.K, put those states back in
    % and renormalize
    if ~isempty(GammaInit) && (size(GammaInit,2) < options.K)
        % States were knocked out, but semisupervised in use, so put them back
        GammaInit = [GammaInit 0.0001*rand(size(GammaInit,1),options.K-size(GammaInit,2))];
        GammaInit = bsxfun(@rdivide,GammaInit,sum(GammaInit,2));
    end

    if isempty(options.hmm) % Initialisation of the hmm 
        % GammaInit is required for obsinit, or for hmmtrain when updateGamma==0
        hmm_wr = struct('train',struct());
        hmm_wr.K = options.K;
        hmm_wr.train = options;
        hmm_wr = hmmhsinit(hmm_wr,GammaInit,T);
        [hmm_wr,residuals_wr] = obsinit(data,T,hmm_wr,GammaInit);
        if isfield(options,'distribution') && strcmp(options.distribution,'logistic')
            residuals_wr = getresidualslogistic(data.X,T,options.logisticYdim); 
        end
    else % using a warm restart from a previous run
        hmm_wr = versCompatibilityFix(options.hmm);
        options = rmfield(options,'hmm');
        train = hmm_wr.train; 
        hmm_wr.train = options;
        hmm_wr.train.active = train.active;  
        % set priors
        Dir2d_alpha = hmm_wr.Dir2d_alpha; Dir_alpha = hmm_wr.Dir_alpha; P = hmm_wr.P; Pi = hmm_wr.Pi;
        if isfield(hmm_wr,'prior') && isfield(hmm_wr.prior,'Omega'), Omega_prior = hmm_wr.prior.Omega; end
        if isfield(hmm_wr,'prior'), hmm_wr = rmfield(hmm_wr,'prior'); end
        hmm_wr = hmmhsinit(hmm_wr); 
        hmm_wr.Dir2d_alpha = Dir2d_alpha; hmm_wr.Dir_alpha = Dir_alpha; hmm_wr.P = P; hmm_wr.Pi = Pi; 
        if exist('Omega_prior','var'), hmm_wr.prior.Omega = Omega_prior; end
        % get residuals
        if ~isfield(options,'distribution') || ~strcmp(options.distribution,'logistic')
            residuals_wr = getresiduals(data.X,T,hmm_wr.train.Sind,hmm_wr.train.maxorder,hmm_wr.train.order,...
                hmm_wr.train.orderoffset,hmm_wr.train.timelag,hmm_wr.train.exptimelag,hmm_wr.train.zeromean);
        elseif do_HMM_pca
            residuals_wr = [];
        else
            residuals_wr = getresidualslogistic(data.X,T,options.logisticYdim); 
        end
    end
    
    if hmm_wr.train.tudamonitoring
        hmm_wr.tudamonitor = struct();
        hmm_wr.tudamonitor.synch = zeros(hmm_wr.train.cyc+1,T(1)-1);
        hmm_wr.tudamonitor.accuracy = zeros(hmm_wr.train.cyc+1,T(1)-1);
        sy = getSynchronicity(GammaInit,T);
        hmm_wr.tudamonitor.synch(1,:) = sy;
        which_x = (hmm_wr.train.S(1,:) == -1);
        which_y = (hmm_wr.train.S(1,:) == 1);
        hmm_wr.tudamonitor.accuracy(1,:) = ...
            getAccuracy(residuals_wr(:,which_x),residuals_wr(:,which_y),T,GammaInit,[],0);
        if ~isempty(hmm_wr.train.behaviour)
            fs = fields(hmm_wr.train.behaviour);
            hmm_wr.tudamonitor.behaviour = struct();
            for ifs = 1:length(fs)
                y = hmm_wr.train.behaviour.(fs{ifs});
                f = getBehAssociation(GammaInit,y,T,sy);
                hmm_wr.tudamonitor.behaviour.(fs{ifs}) = f;
            end
        end        
    end
    
    fehist = Inf; 
    for it = 1:options.repetitions
        hmm0 = hmm_wr;
        [hmm0,Gamma0,Xi0,fehist0] = hmmtrain(data,T,hmm0,GammaInit,residuals_wr,fehistInit);
        if options.updateGamma && (fehist0(end)<fehist(end))
            fehist = fehist0; hmm = hmm0;
            residuals = residuals_wr; Gamma = Gamma0; Xi = Xi0;
        elseif ~options.updateGamma
            fehist = []; hmm = hmm0;
            residuals = []; Gamma = GammaInit; Xi = [];
        end
    end
    
    if options.repetitions == 0
        hmm0 = hmm_wr;
        hmm0.train.updateObs = 0;
        [~,Gamma,Xi] = hmmtrain(data,T,hmm0,GammaInit,residuals_wr,fehistInit);
        fehist = []; hmm = hmm0;
        residuals = []; 
    end
    
    if options.decodeGamma && nargout >= 4
        
        vpath = hmmdecode(data.X,T,hmm,1,residuals,0);
        if ~options.keepS_W
            for i=1:hmm.K
                hmm.state(i).W.S_W = [];
            end
        end
    else
        vpath = ones(size(Gamma,1),1);
    end
    hmm.train = rmfield(hmm.train,'Sind');
    
    feterms = []; rho = [];
    
end

if isfield(hmm,'grouping')
    hmm.train = rmfield(hmm.train,'grouping');
end
status = checkGamma(Gamma,T,hmm.train);
if status==1
    warning(['It seems that the inference was trapped in a local minima; ' ...
        'you might want to increment DirichletDiag and rerun'])
end

if gatherStats==1
    hmm.train.DirStats = DirStats; 
    profile off
    profsave(profile('info'),hmm.train.DirStats)
end

end
