function [hmm, Gamma, Xi, vpath, GammaInit, residuals, fehist, feterms, rho] = ...
    hmmmar (data,T,options)
% Main function to train the HMM-MAR model, compute the Viterbi path and,
% if requested, obtain the cross-validated sum of prediction quadratic errors.
%
% INPUT
% data          observations, either a struct with X (time series) and C (classes, optional)
%                             or just a matrix containing the time series
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
    N = numel(cell2mat(T));
else
    N = length(T);
end

% is this going to be using the stochastic learning scheme? 
stochastic_learn = isfield(options,'BIGNbatch') && ...
    (options.BIGNbatch < N && options.BIGNbatch > 0);
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
    warning('In order to use stochastic learning, BIGNbatch needs to be specified')
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
    if ~iscell(data)
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
    if isfield(options,'crosstermsonly') && options.crosstermsonly
        options = checksoptions(options,T,data);
    else
        options = checksoptions(options,T);
    end
else % data can be a cell or a matrix
    if iscell(T)
        for i = 1:length(T)
            if size(T{i},1)==1, T{i} = T{i}'; end
        end
        if size(T,1)==1, T = T'; end
        T = cell2mat(T);
    end
    checkdatacell;
    [options,data] = checkoptions(options,data,T,0);
end

ver = version('-release');
oldMatlab = ~isempty(strfind(ver,'2010')) || ~isempty(strfind(ver,'2010')) ...
    || ~isempty(strfind(ver,'2011')) || ~isempty(strfind(ver,'2012'));

% set the matlab parallel computing environment
if options.useParallel==1 && usejava('jvm')
    if oldMatlab
        if matlabpool('size')==0
            matlabpool
        end
    else
        gcp;
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
                options.filter,options.Fs);
            options.pca_spatial = size(options.As,2);
        end
    else 
        options.As = [];
    end    
    % get PCA loadings 
    if length(options.pca) > 1 || (options.pca > 0 && options.pca ~= 1)
        if ~isfield(options,'A')
            options.A = highdim_pca(data,T,options.pca,...
                options.embeddedlags,options.standardise,...
                options.onpower,options.varimax,options.detrend,...
                options.filter,options.Fs,options.As);
            options.pca = size(options.A,2);
        end
        options.ndim = size(options.A,2);
        options.S = ones(options.ndim);
        orders = formorders(options.order,options.orderoffset,options.timelag,options.exptimelag); 
        options.Sind = formindexes(orders,options.S);
        if ~options.zeromean, options.Sind = [true(1,size(options.Sind,2)); options.Sind]; end
        if isfield(options,'state') && isfield(options.state(1),'train')
            for k = 1:options.K
                options.state(k).train.S = options.S;
                options.state(k).train.Sind = options.Sind;
                options.state(k).train.ndim = options.ndim;
            end
        end
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
    else % hmm unspecified 
        GammaInit = options.Gamma;
        options = rmfield(options,'Gamma');
        [hmm,info] = hmmsinitg(data,T,options,GammaInit);
    end
    if options.BIGcyc>1
        [hmm,fehist,feterms,rho] = hmmstrain(data,T,hmm,info,options);
    else
        fehist = []; feterms = []; rho = [];
    end
    Gamma = []; Xi = []; vpath = []; residuals = [];
    if options.BIGcomputeGamma && nargout >= 2
       [Gamma,Xi] = hmmdecode(data,T,hmm,0); 
    end
    if options.BIGdecodeGamma && nargout >= 4
       vpath = hmmdecode(data,T,hmm,1); 
    end
    
else
    
    % Filtering
    if ~isempty(options.filter)
       data = filterdata(data,T,options.Fs,options.filter);
    end
    % Detrend data
    if options.detrend
       data = detrenddata(data,T); 
    end
    % Standardise data and control for ackward trials
    data = standardisedata(data,T,options.standardise); 
    % Hilbert envelope
    if options.onpower
       data = rawsignal2power(data,T); 
    end
    % pre-embedded PCA transform
    if length(options.pca_spatial) > 1 || (options.pca_spatial > 0 && options.pca_spatial ~= 1)
        if isfield(options,'As')
            data.X = bsxfun(@minus,data.X,mean(data.X));   
            data.X = data.X * options.As; 
        else
            [options.As,data.X] = highdim_pca(data.X,T,options.pca_spatial,0,0,0,0);
            options.pca_spatial = size(options.As,2);
        end
    end    
    % Embedding
    if length(options.embeddedlags) > 1  
        [data,T] = embeddata(data,T,options.embeddedlags);
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
        orders = formorders(options.order,options.orderoffset,options.timelag,options.exptimelag);
        options.Sind = formindexes(orders,options.S);
        if ~options.zeromean, options.Sind = [true(1,size(options.Sind,2)); options.Sind]; end
        if isfield(options,'state') && isfield(options.state(1),'train')
            for k = 1:options.K
                options.state(k).train.S = options.S;
                options.state(k).train.Sind = options.Sind;
                options.state(k).train.ndim = options.ndim;
            end
        end
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
    
    % Data transformation for crosstermsonly==1
    options.ndim = size(data.X,2);
    if options.crosstermsonly
        Ttmp = T;
        T = T + 1;
        X = zeros(sum(T),2*options.ndim);
        for n=1:N
            t1 = (1:T(n)) + sum(T(1:n-1));
            t2 = (1:Ttmp(n)) + sum(Ttmp(1:n-1));
            X(t1(2:end),1:options.ndim) = data.X(t2,:);
            X(t1(1:end-1),(options.ndim+1):end) = data.X(t2,:);
        end
        data.X = X; clear X
        options.ndim = 2 * options.ndim;
    end
    options.crosstermsonly = 0; % so that recursive calls to hmmmar do not mess up

    if isempty(options.Gamma) && isempty(options.hmm) % both unspecified
        if options.K > 1
            Sind = options.Sind;
            if options.initrep>0 && ...
                    (strcmpi(options.inittype,'HMM-MAR') || strcmpi(options.inittype,'HMMMAR'))
                GammaInit = hmmmar_init(data,T,options,Sind);
            elseif options.initrep>0 &&  strcmpi(options.inittype,'EM')
                error('EM init is deprecated; use HMM-MAR initialisation instead')
                %options.nu = sum(T)/200;
                %options.Gamma = em_init(data,T,options,Sind);
            elseif options.initrep>0 && strcmpi(options.inittype,'GMM')
                error('GMM init is deprecated; use HMM-MAR initialisation instead')
                %options.Gamma = gmm_init(data,T,options);
            elseif strcmpi(options.inittype,'random') || options.initrep==0
                GammaInit = initGamma_random(T-options.maxorder,options.K,options.DirichletDiag);
            else
                error('Unknown init method')
            end
        else
            options.Gamma = ones(sum(T)-length(T)*options.maxorder,1);
            GammaInit = options.Gamma;
        end
    elseif isempty(options.Gamma) && ~isempty(options.hmm) % Gamma unspecified
        GammaInit = [];
    else % hmm unspecified, or both specified
        GammaInit = options.Gamma;
    end
    options = rmfield(options,'Gamma');

    % If initialization Gamma has fewer states than options.K, put those states back in
    % and renormalize
    if size(GammaInit,2) < options.K 
        % States were knocked out, but semisupervised in use, so put them back
        GammaInit = [GammaInit 0.0001*rand(size(GammaInit,1),options.K-size(GammaInit,2))];
        GammaInit = bsxfun(@rdivide,GammaInit,sum(GammaInit,2));
    end

    fehist = Inf;
    if isempty(options.hmm) % Initialisation of the hmm 
        % GammaInit is required for obsinit, or for hmmtrain when updateGamma==0
        hmm_wr = struct('train',struct());
        hmm_wr.K = options.K;
        hmm_wr.train = options;
        %if options.whitening, hmm_wr.train.A = A; hmm_wr.train.iA = iA;  end
        hmm_wr = hmmhsinit(hmm_wr);
        [hmm_wr,residuals_wr] = obsinit(data,T,hmm_wr,GammaInit);
    else % using a warm restart from a previous run
        hmm_wr = versCompatibilityFix(options.hmm);
        options = rmfield(options,'hmm');
        hmm_wr.train = options;
        residuals_wr = getresiduals(data.X,T,hmm_wr.train.Sind,hmm_wr.train.maxorder,hmm_wr.train.order,...
            hmm_wr.train.orderoffset,hmm_wr.train.timelag,hmm_wr.train.exptimelag,hmm_wr.train.zeromean);
    end
    
    for it=1:options.repetitions
        hmm0 = hmm_wr;
        residuals0 = residuals_wr;
        [hmm0,Gamma0,Xi0,fehist0] = hmmtrain(data,T,hmm0,GammaInit,residuals0,options.fehist);
        if options.updateGamma==1 && fehist0(end)<fehist(end)
            fehist = fehist0; hmm = hmm0;
            residuals = residuals0; Gamma = Gamma0; Xi = Xi0;
        elseif options.updateGamma==0
            fehist = []; hmm = hmm0;
            residuals = []; Gamma = GammaInit; Xi = [];
        end
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

hmm.train = rmfield(hmm.train,'grouping'); 
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
