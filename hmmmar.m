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
    for i = 1:length(T)
        if size(T{i},1)==1, T{i} = T{i}'; end
    end
    N = numel(cell2mat(T));
else
    N = length(T);
end

stochastic_learn = isfield(options,'BIGNbatch') && ...
    (options.BIGNbatch < N && options.BIGNbatch > 0);
options = checkspelling(options);

if stochastic_learn, % data is a cell, either with strings or with matrices
    if ~iscell(data)
       dat = cell(N,1); TT = cell(N,1);
       for i=1:N
          t = 1:T(i);
          dat{i} = data(t,:); TT{i} = T(i);
          try data(t,:) = []; 
          catch, error('The dimension of data does not correspond to T');
          end
       end
       if ~isempty(data), 
           error('The dimension of data does not correspond to T');
       end 
       data = dat; T = TT; clear dat TT
    end
    options = checksoptions(options,T);
else % data is a struct, with a matrix .X  
    if iscell(data)
        if size(data,1)==1, data = data'; end
        data = cell2mat(data);
    end
    if iscell(T)
        if size(T,1)==1, T = T'; end
        T = cell2mat(T);
    end
    [options,data] = checkoptions(options,data,T,0);
    if options.standardise == 1
        for i = 1:N
            t = (1:T(i)) + sum(T(1:i-1));
            data.X(t,:) = data.X(t,:) - repmat(mean(data.X(t,:)),length(t),1);
            sdx = std(data.X(t,:));
            if any(sdx==0)
                error('At least one of the trials/segments/subjects has variance equal to zero');
            end
            data.X(t,:) = data.X(t,:) ./ repmat(sdx,length(t),1);
        end
    else
        for i = 1:N
            t = (1:T(i)) + sum(T(1:i-1));
            if any(std(data.X(t,:))==0)
                error('At least one of the trials/segments/subjects has variance equal to zero');
            end
        end
    end
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
    
    % get PCA loadings 
    if length(options.pca) > 1 || options.pca > 0 
        if ~isfield(options,'A')
            options.A = highdim_pca(data,T,options.pca,options.embeddedlags,options.standardise);
        end
        options.ndim = size(options.A,2);
        options.S = ones(options.ndim);
        orders = formorders(options.order,options.orderoffset,options.timelag,options.exptimelag); 
        options.Sind = formindexes(orders,options.S);
    end
    if options.pcamar > 0 && ~isfield(options,'B')
        options.B = pcamar_decomp(data,T,options);
    end
    if options.pcapred > 0 && ~isfield(options,'V')
        options.V = pcapred_decomp(data,T,options);
    end
    
    if isempty(options.Gamma) && isempty(options.hmm)
        [hmm,info] = hmmsinit(data,T,options);
        GammaInit = []; 
    elseif isempty(options.Gamma) && ~isempty(options.hmm)
        hmm = versCompatibilityFix(options.hmm);
        GammaInit = [];
        [hmm,info] = hmmsinith(data,T,options,hmm);
    else % ~isempty(options.Gamma)
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
       [Gamma,Xi] = hmmdecode(data,T,hmm,0,[],[]); 
    end
    if options.BIGdecodeGamma && nargout >= 4
       vpath = hmmdecode(data,T,hmm,1,[],[]); 
    end
    
else
    
    % embed data?
    if length(options.embeddedlags) > 1  
        [data,T] = embeddata(data,T,options.embeddedlags);
    end
    % pca
    if length(options.pca) > 1 || options.pca > 0  
        if isfield(options,'A')
            data.X = data.X - repmat(mean(data.X),mean(data.X,1),1);
            data.X = data.X * options.A; 
        else
            [options.A,data.X] = highdim_pca(data.X,T,options.pca,0,0);
        end
        if options.standardise_pc == 1
            for i = 1:N
                t = (1:T(i)) + sum(T(1:i-1));
                data.X(t,:) = data.X(t,:) - repmat(mean(data.X(t,:)),length(t),1);
                data.X(t,:) = data.X(t,:) ./ repmat(std(data.X(t,:)),length(t),1);
            end
        end
        options.ndim = size(options.A,2);
        options.S = ones(options.ndim);
        orders = formorders(options.order,options.orderoffset,options.timelag,options.exptimelag);
        options.Sind = formindexes(orders,options.S);
    end
    if options.pcamar > 0 && ~isfield(options,'B')
        options.B = pcamar_decomp(data,T,options);
    end
    if options.pcapred > 0 && ~isfield(options,'V')
        options.V = pcapred_decomp(data,T,options);
    end    
    options.ndim = size(data.X,2);

    if isempty(options.Gamma) && isempty(options.hmm)
        if options.K > 1
            Sind = options.Sind;
            if options.initrep>0 && ...
                    (strcmpi(options.inittype,'HMM-MAR') || strcmpi(options.inittype,'HMMMAR'))
                options.Gamma = hmmmar_init(data,T,options,Sind);
            elseif options.initrep>0 &&  strcmpi(options.inittype,'EM')
                error('EM init is deprecated; use HMM-MAR initialisation instead')
                %options.nu = sum(T)/200;
                %options.Gamma = em_init(data,T,options,Sind);
            elseif options.initrep>0 && strcmpi(options.inittype,'GMM')
                error('GMM init is deprecated; use HMM-MAR initialisation instead')
                %options.Gamma = gmm_init(data,T,options);
            elseif strcmpi(options.inittype,'random') || options.initrep==0
                options.Gamma = initGamma_random(T-options.maxorder,options.K,options.DirichletDiag);
            else
                error('Unknown init method')
            end
        else
            options.Gamma = ones(sum(T)-length(T)*options.maxorder,1);
        end
        GammaInit = options.Gamma;
        options = rmfield(options,'Gamma');
    elseif isempty(options.Gamma) && ~isempty(options.hmm)
        GammaInit = [];
    else % ~isempty(options.Gamma)
        GammaInit = options.Gamma;
        options = rmfield(options,'Gamma');
    end

    % If initialization Gamma has fewer states than options.K, put those states back in
    % and renormalize
    if size(GammaInit,2) < options.K 
        % States were knocked out, but semisupervised in use, so put them back
        GammaInit = [GammaInit 0.0001*rand(size(GammaInit,1),options.K-size(GammaInit,2))];
        GammaInit = bsxfun(@rdivide,GammaInit,sum(GammaInit,2));
    end

    fehist = Inf;
    if isempty(options.hmm) % Initialisation of the hmm
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
        if options.updateGamma==1 && fehist0(end)<fehist(end),
            fehist = fehist0; hmm = hmm0;
            residuals = residuals0; Gamma = Gamma0; Xi = Xi0;
        elseif options.updateGamma==0,
            fehist = []; hmm = hmm0;
            residuals = []; Gamma = GammaInit; Xi = [];
        end
    end
    
    if options.decodeGamma && nargout >= 4
        vpath = hmmdecode(data.X,T,hmm,1,residuals);
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

if gatherStats==1
    hmm.train.DirStats = DirStats; 
    profile off
    profsave(profile('info'),hmm.train.DirStats)
end

if options.pca > 0
    hmm.train.A = options.A; 
end
    
end
