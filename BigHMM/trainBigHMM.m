function [metastates,markovTrans,prior,fehist,feterms] = ...
    trainBigHMM(files,T,options)
% 1) Initialize BigHMM by estimating a separate HMM on each file,
%    and then using an adapted version of kmeans to clusterize the states
%    (see initBigHMM)
% 2) Run the BigHMM algorithm
%
% INPUTS
% files: cell with strings referring to the subject files
% T: cell of vectors, where each element has the length of each trial per
%       subject. Dimension of T{n} has to be (1 x nTrials)
% options: HMM options for both the subject and the group runs
%
% COVTYPE=UNIQUEFULL/UNIQUEDIAG NOT COMPLETELY IMPLEMENTED!!
% Diego Vidaurre, OHBA, University of Oxford (2015)

N = length(files);

if ~isfield(options,'K'), error('K was not specified');
else K = options.K; end
% Specific BigHMM options
if ~isfield(options,'BIGNbatch'), BIGNbatch = 10;
else BIGNbatch = options.BIGNbatch; end
if ~isfield(options,'uniqueTrans'), uniqueTrans = 0;
else uniqueTrans = options.uniqueTrans; end
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
if ~isfield(options,'initcyc'), options.initcyc = 50; end
if ~isfield(options,'initrep'), options.initrep = 3; end
options.dropstates = 0; 
options.verbose = 0; % shut up the individual hmmmar output

if options.order>0
    [options.orders,options.order] = formorders(options.order,options.orderoffset,options.timelag,options.exptimelag);
else
    options.orders = [];
end
npred = length(options.orders) + (~options.zeromean);

if strcmp(options.covtype,'full') || strcmp(options.covtype,'diag')
    BIGcovtype = 'subject';
else
    if options.zeromean==1
        error('Neither states'' mean nor covariance are allow to vary')
    end
    BIGcovtype = 'unique';
end
    
X = loadfile(files{1}); ndim = size(X,2); sumT = 0;
subjfe_init = zeros(N,3);
loglik_init = zeros(N,1);
statekl_init = zeros(K,1);
subjfe = zeros(N,3,1);
loglik = zeros(N,1);
statekl = zeros(K,1);

% Initialization
if ~isfield(options,'metastates')
    
    % init sufficient statistics
    
    subj_m_init = zeros(npred,ndim,N,K);
    subj_gram_init = zeros(npred,npred,N,K);
    if strcmp(options.covtype,'diag')
        subj_err_init = zeros(ndim,N,K); gram_init = [];
    elseif strcmp(options.covtype,'full')
        subj_err_init = zeros(ndim,ndim,N,K); gram_init = [];
    elseif strcmp(options.covtype,'uniquediag')
        gram_init = zeros(1,ndim); subj_err_init = [];
    else % uniquefull
        gram_init = zeros(ndim,ndim); subj_err_init = [];
    end
    subj_time_init = zeros(N,K);
    
    % init subject parameters
    P_init = zeros(K,K,N); Pi_init = zeros(K,N);
    Dir2d_alpha_init = zeros(K,K,N); Dir_alpha_init = zeros(K,N);
    
    best_fe = Inf;
    for cycle = 1:BIGinitcyc
        
        % train individual HMMs
        I = randperm(N);
        for ii = 1:length(I)
            % read data
            i = I(ii);
            X = loadfile(files{i});
            XX = formautoregr(X,T{i},options.orders,options.order,options.zeromean);
            sumT = sumT + sum(T{i});
            if cycle==1
                if ii==1
                    range_data = range(X);
                else
                    range_data = max(range_data,range(X));
                end
            end
            % Running the individual HMM
            [hmm_i,Gamma,Xi] = hmmmar(X,T{i},options); 
            if uniqueTrans && ii==1
                Dir2d_alpha_prior = hmm_i.prior.Dir2d_alpha;
                Dir_alpha_prior = hmm_i.prior.Dir_alpha;
            end
            if BIGverbose
                fprintf('Init run %d, subject %d \n',cycle,ii);
            end
            if uniqueTrans % one P/Pi for all subjects
                for trial=1:length(T{i})
                    t = sum(T{i}(1:trial-1)) - options.order*(trial-1) + 1;
                    Dir_alpha_init(:,i) = Dir_alpha_init(:,i) + Gamma(t,:)';
                end
                Dir2d_alpha_init(:,:,i) = squeeze(sum(Xi,1));
                [P_init,Pi_init] = computePandPi(Dir_alpha_init,Dir2d_alpha_init);
            else
                P_init(:,:,i) = hmm_i.P; Pi_init(:,i) = hmm_i.Pi';
                Dir2d_alpha_init(:,:,i) = hmm_i.Dir2d_alpha; 
                Dir_alpha_init(:,i) = hmm_i.Dir_alpha';                
            end
            K_i = length(hmm_i.state);
            % Reassigning ordering of the states according the closest metastates
            if ii==1
                assig = 1:K_i;
                if K_i<K, 
                    warning('The first HMM run needs to return K states, you might want to start again..\n')
                end
            else
                dist = Inf(K_i,K);
                for j = 1:K_i
                    XG_j = XX' .* repmat(Gamma(:,j)',size(XX,2),1);
                    metastate_j = metastate_new( ...
                        hmm_i.state(j).Omega.Gam_rate',...
                        hmm_i.state(j).Omega.Gam_shape,...
                        XG_j * XX + 0.01 * eye(size(XX,2)),...
                        XG_j * X);
                    for k = 1:K
                        dist(j,k) = symm_kl_div(metastate_j, metastates_init(k));
                    end
                end
                assig = munkres(dist); % linear assignment problem
            end
            % update sufficient statistics
            for k=1:K_i,
                XG = XX' .* repmat(Gamma(:,k)',size(XX,2),1);
                subj_m_init(:,:,i,assig(k)) = XG * X;
                subj_gram_init(:,:,i,assig(k)) = XG * XX;  
                if strcmp(options.covtype,'diag') 
                    % this is counting the prior N times more than it should
                    subj_err_init(:,i,assig(k)) = hmm_i.state(k).Omega.Gam_rate';
                elseif strcmp(options.covtype,'full') 
                    subj_err_init(:,:,i,assig(k)) = hmm_i.state(k).Omega.Gam_rate;
                end
                subj_time_init(i,assig(k)) = hmm_i.state(k).Omega.Gam_shape;
                % cov mats: note also that these are the individual ones, and,
                % hence, an underestimation of the group ones
            end
            if strcmp(options.covtype,'uniquefull') || strcmp(options.covtype,'uniquediag')
                gram_init = gram_init + hmm_i.Omega.Gam_rate;
            end
            % updating the metastates
            for k = 1:K_i
                if strcmp(options.covtype,'full')
                    metastates_init(k) = metastate_new( ...
                        sum(subj_err_init(:,:,I(1:ii),k),3),...
                        sum(subj_time_init(I(1:ii),k)),...
                        sum(subj_gram_init(:,:,I(1:ii),k),3) + 0.01 * eye(size(XX,2)),...
                        sum(subj_m_init(:,:,I(1:ii),k),3));
                elseif strcmp(options.covtype,'diag')
                    metastates_init(k) = metastate_new( ...
                        sum(subj_err_init(:,I(1:ii),k),2),...
                        sum(subj_time_init(I(1:ii),k)),...
                        sum(subj_gram_init(:,:,I(1:ii),k),3) + 0.01 * eye(size(XX,2)),...
                        sum(subj_m_init(:,:,I(1:ii),k),3));
                end
            end
        end
        
        % set prior
        if cycle==1 
            if isempty(options.BIGprior)
                prior = struct();
                prior.Omega = struct();
                if strcmp(options.covtype,'uniquediag') || strcmp(options.covtype,'diag')
                    prior.Omega.Gam_shape = 0.5 * (ndim+0.1-1);
                    prior.Omega.Gam_rate = 0.5 * range_data;
                else
                    prior.Omega.Gam_shape = ndim+0.1-1;
                    prior.Omega.Gam_rate = diag(range_data);
                end
                prior.Mean = struct();
                prior.Mean.S = (range_data/2).^2;
                prior.Mean.iS = 1 ./ prior.Mean.S;
                prior.Mean.Mu = zeros(1,ndim);
                if ~isempty(options.orders)
                    prior.sigma = struct();
                    prior.sigma.Gam_shape = 0.1*ones(ndim,ndim); %+ 0.05*eye(ndim);
                    prior.sigma.Gam_rate = 0.1*ones(ndim,ndim);%  + 0.05*eye(ndim);
                    prior.alpha = struct();
                    prior.alpha.Gam_shape = 0.1;
                    prior.alpha.Gam_rate = 0.1*ones(1,length(orders));
                end
            else
                prior = options.BIGprior;
            end
        end
        
        % Compute Gamma to get an
        % unbiased group estimation of the metastate covariance matrices;
        % obtaining subject parameters and computing free energy
        metastates_init = adjSw_in_metastate(metastates_init); % adjust the dim of S_W
        if uniqueTrans
            Dir2d_alpha_init_pre = sum(Dir2d_alpha_init) + Dir2d_alpha_prior;  
            Dir_alpha_init_pre = sum(Dir_alpha_init,2)' + Dir_alpha_prior;
            [P_init_pre,Pi_init_pre] = computePandPi(Dir_alpha_init_pre,Dir2d_alpha_init_pre);
        end
        for i = 1:N
            X = loadfile(files{i});
            data = struct('X',X,'C',NaN(sum(T{i})-length(T{i})*options.order,K));
            if uniqueTrans
                hmm = loadhmm(hmm_i,T{i},K,metastates_init,...
                    P_init_pre,Pi_init_pre,Dir2d_alpha_init_pre,Dir_alpha_init_pre,gram_init,prior);
            else
                hmm = loadhmm(hmm_i,T{i},K,metastates_init,...
                    P_init(:,:,i),Pi_init(:,i)',Dir2d_alpha_init(:,:,i),Dir_alpha_init(:,i)',...
                    gram_init,prior);                
            end
            [Gamma,~,Xi,l] = hsinference(data,T{i},hmm,[]);
            hmm = hsupdate(Xi,Gamma,T{i},hmm);
            if uniqueTrans
                for trial=1:length(T{i})
                    t = sum(T{i}(1:trial-1)) - options.order*(trial-1) + 1;
                    Dir_alpha_init(:,i) = Dir_alpha_init(:,i) + Gamma(t,:)';
                end
                Dir2d_alpha_init(:,:,i) = squeeze(sum(Xi,1));
            else
                P_init(:,:,i) = hmm.P; Pi_init(:,i) = hmm.Pi'; % one per subject, not like pure group HMM
                Dir2d_alpha_init(:,:,i) = hmm.Dir2d_alpha; Dir_alpha_init(:,i) = hmm.Dir_alpha';
            end
            for k=1:K
                if ~isempty(options.orders) || (~options.zeromean)
                    E = X - XX * metastates_init(k).W.Mu_W; % using the current mean estimation
                else
                    E = X; 
                end
                if strcmp(options.covtype,'full')
                    subj_err_init(:,:,i,k) = ((E' .* repmat(Gamma(:,k)',size(E,2),1)) * E);
                elseif strcmp(options.covtype,'diag')
                    subj_err_init(:,i,k) = ( sum( (E.^2) .* repmat(Gamma(:,k),1,size(E,2)) ) )';
                end
                subj_time_init(i,k) = sum(Gamma(:,k));
                metastates_init(k).Omega.Gam_rate = sum(subj_err_init(:,:,I(1:ii),k),3);
                metastates_init(k).Omega.Gam_shape = sum(subj_time_init(I(1:ii),k));
            end
            subjfe_init(i,1) = - GammaEntropy(Gamma,Xi,T{i},0); 
            subjfe_init(i,2) = - GammaavLL(hmm,Gamma,Xi,T{i});
            if ~uniqueTrans
                subjfe_init(i,3) = + KLtransition(hmm);
            end
            loglik_init(i) = sum(l);
        end
        if uniqueTrans
            hmm.Dir_alpha = sum(Dir_alpha_init,2)' + Dir_alpha_prior; 
            hmm.Dir2d_alpha = sum(Dir2d_alpha_init,3) + Dir2d_alpha_prior;
            [P_init,Pi_init] = computePandPi(hmm.Dir_alpha_init,hmm.Dir2d_alpha_init);
            subjfe_init(:,3) = + KLtransition(hmm) / N; % "share" the common KL
        end
        for k = 1:K 
            statekl_init(k) = KLstate(metastates_init(k),prior,options);
        end
        fe = - sum(loglik_init) + sum(subjfe_init(:)) + sum(statekl_init);
        
        if fe<best_fe
            best_fe = fe;
            metastates = metastates_init;
            gramm = gram_init;
            hmm0 = hmm_i;
            subjfe(:,:,1) = subjfe_init(:,:,1);
            loglik(:,1) = loglik_init; 
            statekl(:,1) = statekl_init;
            P = P_init; Pi = Pi_init;
            Dir2d_alpha = Dir2d_alpha_init; Dir_alpha = Dir_alpha_init;
            fehist = best_fe;
        end
        
        if BIGverbose
            fprintf('Init run %d, FE=%g (best=%g) \n',cycle,fe,best_fe);
        end
        
    end
    
    if BIGverbose
        fprintf('Cycle 1, free energy: %g \n',fehist);
    end
    
else % initial metastates specified by the user
    
    metastates = options.metastates; options = rmfield(options,'metastates');
    if strcmp(options.covtype,'diag')
        gramm = [];
    elseif strcmp(options.covtype,'full')
        gramm = [];
    elseif strcmp(options.covtype,'uniquediag')
        gramm = zeros(1,ndim);  
    else % uniquefull
        gramm = zeros(ndim,ndim);  
    end
       
    P = zeros(K,K,N); Pi = zeros(K,N);
    Dir2d_alpha = zeros(K,K,N); Dir_alpha = zeros(K,N);

    % collect some stats
    for i = 1:N
        X = loadfile(files{i});
        if i==1
            options.inittype = 'random';
            options.cyc = 1;
            hmm0 = hmmmar(X,T{i},options);
            range_data = range(X);
        else
            range_data = max(range_data,range(X));
        end
        if strcmp(options.covtype,'uniquefull')
            gramm = gramm + X' * X;
        elseif strcmp(options.covtype,'uniquediag')
            gramm = gramm + sum(X.^2);
        end
    end
    
    % set prior
    prior = struct();
    prior.Omega = struct();
    if strcmp(options.covtype,'uniquediag') || strcmp(options.covtype,'diag')
        prior.Omega.Gam_shape = 0.5 * (ndim+0.1-1);
        prior.Omega.Gam_rate = 0.5 * range_data;
    else
        prior.Omega.Gam_shape = ndim+0.1-1;
        prior.Omega.Gam_rate = diag(range_data);
    end
    prior.Mean = struct();
    prior.Mean.S = (range_data/2).^2;
    prior.Mean.iS = 1 ./ prior.Mean.S;
    prior.Mean.Mu = zeros(1,ndim);
    if ~isempty(options.orders)
        prior.sigma = struct();
        prior.sigma.Gam_shape = 0.1*ones(ndim,ndim); %+ 0.05*eye(ndim);
        prior.sigma.Gam_rate = 0.1*ones(ndim,ndim);%  + 0.05*eye(ndim);
        prior.alpha = struct();
        prior.alpha.Gam_shape = 0.1;
        prior.alpha.Gam_rate = 0.1*ones(1,length(orders));
    end
    
    % Init subject models and free energy computation
    for i = 1:N
        X = loadfile(files{i});
        data = struct('X',X,'C',NaN(size(X,1),K));
        hmm = loadhmm(hmm0,T{i},K,metastates,[],[],[],[],gramm,prior);
        % get gamma
        [Gamma,~,Xi,l] = hsinference(data,T{i},hmm,[]);
        % compute transition prob
        hmm = hsupdate(Xi,Gamma,T{i},hmm);
        if uniqueTrans
            for trial=1:length(T{i})
                t = sum(T{i}(1:trial-1)) - options.order*(trial-1) + 1;
                Dir_alpha(:,i) = Dir_alpha(:,i) + Gamma(t,:)';
            end
            Dir2d_alpha(:,:,i) = squeeze(sum(Xi,1));
        else
            P(:,:,i) = hmm.P; Pi(:,i) = hmm.Pi'; % one per subject, not like pure group HMM
            Dir2d_alpha(:,:,i) = hmm.Dir2d_alpha; Dir_alpha(:,i) = hmm.Dir_alpha';
        end
        % compute free energy
        loglik(i,1) = sum(l);  
        subjfe(i,1,1) = - GammaEntropy(Gamma,Xi,T{i},0); 
        subjfe(i,2,1) = - GammaavLL(hmm,Gamma,Xi,T{i});
        if ~uniqueTrans
            subjfe(i,3,1) = + KLtransition(hmm);
        end
    end
    if uniqueTrans
        hmm.Dir_alpha = sum(Dir_alpha,2)' + Dir_alpha_prior;
        hmm.Dir2d_alpha = sum(Dir2d_alpha,3) + Dir2d_alpha_prior;
        [P,Pi] = computePandPi(hmm.Dir_alpha,hmm.Dir2d_alpha);
        subjfe(:,3,1) = KLtransition(hmm) / N; % "share" the common KL
    end
    for k = 1:K
        statekl(k,1) = KLstate(metastates(k),prior,options.covtype,options.zeromean);
    end
    fehist = sum(- loglik(:,1) + sum(sum(subjfe(:,:,1))) + sum(statekl(:,1)));
    
    if BIGverbose
        fprintf('Cycle 1, free energy: %g \n',fehist);
    end
  
end

clear metastate_gamma_init metastate_m_init metastate_gram_init metastates_init 
clear gram_init subjfe_init loglik_init statekl_init Pi_init P_init Dir2d_alpha_init Dir_alpha_init
    

% init stuff for stochastic learning
nUsed = zeros(1,N);
BIGbase_weights = BIGbase_weights * ones(1,N);
sampling_weights = BIGbase_weights;
undertol = 0;

% load('/tmp/debugBigHMM.mat'); gramm = [];
% fprintf('Init run %d, FE=%g (best=%g) \n',cycle,fe,best_fe);
% % load('/tmp/options_bighmm'); BIGNbatch = options_bighmm.BIGNbatch; 
% BIGcyc = 25; BIGNbatch = N ; % N = 1; 

Tfactor = N/BIGNbatch; 

%[ sum(loglik(:,1)) squeeze(sum(subjfe(:,1,1),1)) squeeze(sum(subjfe(:,2,1),1)) squeeze(sum(subjfe(:,3,1),1))]

%load('/tmp/debugBigHMM.mat')
% Stochastic learning
for cycle = 2:BIGcyc
    
    % sampling batch
    I = datasample(1:N,BIGNbatch,'Replace',false,'Weights',sampling_weights);
    %I = 1:BIGNbatch; %I = 1:N;
    nUsed(I) = nUsed(I) + 1;
    nUsed = nUsed - min(nUsed) + 1;
    sampling_weights = BIGbase_weights.^nUsed;
        
    % read data for this batch
    Tbatch = [];
    for ii = 1:length(I), i = I(ii); Tbatch = [Tbatch T{i}]; end
    X = zeros(sum(Tbatch),ndim); t = 0;
    for ii = 1:length(I)
        i = I(ii);
        X(t+1:t+sum(T{i}),:) = loadfile(files{i});
        t = t + sum(T{i});
    end
    
    % local parameters (Gamma, Xi, P, Pi, Dir2d_alpha and Dir_alpha)
    tacc = 0;
    Gamma = cell(BIGNbatch,1); Xi = cell(BIGNbatch,1);
    for ii = 1:length(I)
        i = I(ii); 
        t = (1:sum(T{i})) + tacc; tacc = tacc + length(t);
        data = struct('X',X(t,:),'C',NaN(sum(T{i})-length(T{i})*options.order,K));
        if uniqueTrans
            hmm = loadhmm(hmm0,T{i},K,metastates,P,Pi,hmm.Dir2d_alpha,hmm.Dir_alpha,gramm,prior);
        else
            hmm = loadhmm(hmm0,T{i},K,metastates,P(:,:,i),Pi(:,i)',Dir2d_alpha(:,:,i),Dir_alpha(:,i)',gramm,prior);
        end
        [Gamma{ii},~,Xi{ii},l] = hsinference(data,T{i},hmm,[]); 
        hmm = hsupdate(Xi{ii},Gamma{ii},T{i},hmm);
        if uniqueTrans
            for trial=1:length(T{i})
                t = sum(T{i}(1:trial-1)) - options.order*(trial-1) + 1;
                Dir_alpha(:,i) = Dir_alpha(:,i) + Gamma(t,:)';
            end
            Dir2d_alpha(:,:,i) = Dir2d_alpha(:,:,i) + squeeze(sum(Xi,1));
        else
            P(:,:,i) = hmm.P; Pi(:,i) = hmm.Pi'; % one per subject, not like pure group HMM
            Dir2d_alpha(:,:,i) = hmm.Dir2d_alpha; Dir_alpha(:,i) = hmm.Dir_alpha';
        end
        subjfe(i,1,cycle) = - GammaEntropy(Gamma{ii},Xi{ii},T{i},0); 
        subjfe(i,2,cycle) = - GammaavLL(hmm,Gamma{ii},Xi{ii},T{i}); 
        if ~uniqueTrans
            subjfe(i,3,cycle) = + KLtransition(hmm);
        end
    end
    if uniqueTrans
        hmm.Dir_alpha = sum(Dir_alpha_init,2)' + Dir_alpha_prior;
        hmm.Dir2d_alpha = sum(Dir2d_alpha_init,3) + Dir2d_alpha_prior;
        [P,Pi] = computePandPi(hmm.Dir_alpha,hmm.Dir2d_alpha);
        subjfe(:,3,cycle) = + KLtransition(hmm) / N; % "share" the common KL
    end
    
    % global parameters (metastates), and collect metastate free energy
    rho = (cycle + BIGdelay)^(-BIGforgetrate); 
    metastates = updateBigOmega(metastates,cell2mat(Gamma),X,Tbatch,Tfactor,rho,prior,options);
    metastates = updateBigW(metastates,cell2mat(Gamma),X,Tbatch,Tfactor,rho,prior,options);
    if options.order>0
        metastates = updateBigSigma(metastates,rho,prior,options);
        metastates = updateBigAlpha(metastates,rho,prior,options);
    end
    for k=1:K
        statekl(k,cycle) = KLstate(metastates(k),prior,options);
    end
    
    % compute likelihood
    %old_ll = loglik(I,cycle);
    loglik(I,cycle) = XavLL(X,T(I),metastates,Gamma,options);
    for i = setdiff(1:N,I)
        loglik(i,cycle) = loglik(i,cycle-1);
        for j = 1:3
            subjfe(i,j,cycle) = subjfe(i,j,cycle-1);
        end
    end
    
    %if ((-sum(loglik(:,cycle))+sum(statekl(:,cycle))) - (-sum(old_ll)+sum(statekl(:,cycle-1)))) > 1e-8  
    %        fprintf('3\n'); pause(0.5)
    %end;  
    
    %[ sum(loglik(:,cycle)) squeeze(sum(subjfe(:,1,cycle),1)) squeeze(sum(subjfe(:,2,cycle),1)) squeeze(sum(subjfe(:,3,cycle),1))]

    fehist = [ fehist; (-sum(loglik(:,cycle)) + sum(statekl(:,cycle)) + sum(sum(subjfe(:,:,cycle)))) ];
    ch = (fehist(end)-fehist(end-1)) / abs(fehist(end)-fehist(1));
    if BIGverbose
        fprintf('Cycle %d, free energy: %g (relative change %g) \n',cycle,fehist(end),ch);
    end
    if cycle>5 && abs(ch) < BIGtol && cycle>BIGmincyc, 
        undertol = undertol + 1; 
    else
        undertol = 0;
    end
    if undertol > BIGundertol_tostop, break; end
    
end

markovTrans = struct();
markovTrans.P = P;
markovTrans.Pi = Pi;
markovTrans.Dir2d_alpha = Dir2d_alpha;
markovTrans.Dir_alpha = Dir_alpha;
markovTrans.prior.Dir2d_alpha = hmm.prior.Dir2d_alpha;
markovTrans.prior.Dir_alpha = hmm.prior.Dir_alpha;

feterms = [sum(loglik(:,end)) sum(subjfe(:,:,end)) sum(statekl(:,end)) ];

end


function metastate = metastate_new(rate,shape,gram,m)
metastate = struct();
metastate.W = struct();
metastate.W.iS_W = gram;
metastate.W.S_W = inv(gram);
metastate.W.Mu_W = metastate.W.S_W * m;
metastate.Omega = struct();
metastate.Omega.Gam_rate = rate;
metastate.Omega.Gam_shape = shape;
if isvector(rate)
    metastate.Omega.Gam_irate = 1 ./ metastate.Omega.Gam_rate;
else
    metastate.Omega.Gam_irate = inv(metastate.Omega.Gam_rate);
end
end


function metastates = adjSw_in_metastate(metastates)
[nprec,ndim] = size(metastates(1).W.Mu_W);
for k = 1:length(metastates)
    iS_W = metastates(k).W.iS_W;
    S_W = metastates(k).W.S_W;
    if isvector(metastates(k).Omega.Gam_rate) 
        metastates(k).W.S_W = zeros(ndim,nprec,nprec);
        metastates(k).W.iS_W = zeros(ndim,nprec,nprec);
        for n=1:ndim
            metastates(k).W.S_W(n,:,:) = S_W;
            metastates(k).W.iS_W(n,:,:) = iS_W;
        end
    else
        metastates(k).W.S_W = zeros(ndim*nprec,ndim*nprec);
        for n = 1:ndim
            ind = (1:nprec) + (n-1)*nprec;
            metastates(k).W.S_W(ind,ind) = S_W;
            metastates(k).W.iS_W(ind,ind) = iS_W;
        end
    end
end
end


function [P,Pi] = computePandPi(Dir_alpha,Dir2d_alpha)
K = length(Dir_alpha);
P = zeros(K); Pi = zeros(1,K);
for j = 1:K
    PsiSum = psi(sum(Dir2d_alpha(j,:)));
    for i = 1:K,
        P(j,i) = exp(psi(Dir2d_alpha(j,i))-PsiSum);
    end;
    P(j,:) = P(j,:) ./ sum(P(j,:));
end
PsiSum = psi(sum(Dir_alpha,2));
for i = 1:K
    hmm.Pi(i) = exp(psi(Dir_alpha(i))-PsiSum);
end
Pi = Pi ./ sum(Pi);
end


function kl = symm_kl_div(m1,m2)
% symmetric kullback leibler divergence between MVN m1.W and m2.W
ndim = size(m1.W.Mu_W,2);
kl = 0;
for n=1:ndim
    kl = 0.5 * gauss_kl(m1.W.Mu_W(:,n),m2.W.Mu_W(:,n),m1.W.S_W,m2.W.S_W) + ...
        0.5 * gauss_kl(m2.W.Mu_W(:,n),m1.W.Mu_W(:,n),m2.W.S_W,m1.W.S_W);
end
end


function metastates = updateBigOmega(metastates,Gamma,X,Tbatch,Tfactor,rho,prior,options)
ndim = size(X,2); K = size(Gamma,2);
orders = options.orders;
XX = formautoregr(X,Tbatch,orders,options.order,options.zeromean);
X = trimX(X,Tbatch,options.order);
for k=1:K
    if strcmp(options.covtype,'diag')
        swx2 = zeros(Tres,ndim);
        if isempty(metastates(k).W.Mu_W)
            e = X.^2;
        else
            e = (X - XX * metastates(k).W.Mu_W).^2;
            for n=1:ndim
                if ndim==1
                    swx2(:,n) = sum(XX * metastates(k).W.S_W .* XX,2);
                else
                    swx2(:,n) = sum(XX * permute(metastates(k).W.S_W,[2 3 1]) .* XX,2);
                end
            end
        end
        rate = prior.Omega.Gam_rate + Tfactor * sum( repmat(Gamma(:,k),1,ndim) .* (e + swx2) ) / 2;
        shape = prior.Omega.Gam_shape + Tfactor * sum(Gamma(:,k)) / 2;
    elseif strcmp(options.covtype,'full')
        swx2 =  zeros(ndim,ndim);
        XXGXX = (XX' .* repmat(Gamma(:,k)',size(XX,2),1)) * XX;
        if isempty(metastates(k).W.Mu_W)
            e = (X' .* repmat(Gamma(:,k)',ndim,1)) * X;
            swx2 = zeros(size(e));
        else
            e = (X - XX * metastates(k).W.Mu_W);
            e = (e' .* repmat(Gamma(:,k)',ndim,1)) * e;
            if isempty(orders)
                swx2 = metastates(k).W.S_W * XXGXX;
            else
                for n1=1:ndim
                    for n2=1:ndim
                        if n2<n1, continue, end;
                        index1 = (0:length(orders)*ndim+(~options.zeromean)-1) * ndim + n1;
                        index2 = (0:length(orders)*ndim+(~options.zeromean)-1) * ndim + n2;
                        index1 = index1(Sind(:,n1)); index2 = index2(Sind(:,n2));
                        swx2(n1,n2) = sum(sum(metastates(k).W.S_W(index1,index2) .* XXGXX));
                        swx2(n2,n1) = swx2(n1,n2);
                    end
                end
            end
        end
        rate = prior.Omega.Gam_rate + Tfactor * (e + swx2);
        shape = prior.Omega.Gam_shape + Tfactor * sum(Gamma(:,k));
    end
    metastates(k).Omega.Gam_rate = (1-rho) * metastates(k).Omega.Gam_rate + rho * rate;
    metastates(k).Omega.Gam_shape = (1-rho) * metastates(k).Omega.Gam_shape + rho * shape;
    if strcmp(options.covtype,'diag')
        metastates(k).Omega.Gam_irate = 1 ./ metastates(k).Omega.Gam_rate;
    elseif strcmp(options.covtype,'full')
        metastates(k).Omega.Gam_irate = inv(metastates(k).Omega.Gam_rate);
    end
end
end


function metastates = updateBigW(metastates,Gamma,X,Tbatch,Tfactor,rho,prior,options)
ndim = size(X,2); K = size(Gamma,2);
orders = options.orders;
XX = formautoregr(X,Tbatch,orders,options.order,options.zeromean);
X = trimX(X,Tbatch,options.order);
for k=1:K
    W = zeros(size(metastates(k).W.Mu_W));
    iSW = zeros(size(metastates(k).W.S_W));  
    if isempty(orders)
        XXGXX = sum(Gamma(:,k));
    else
        XXGXX = (XX' .* repmat(Gamma(:,k)',size(XX,2),1)) * XX;
    end
    if strcmp(options.covtype,'full'), Xdiv = XXGXX \ XX'; end
    if strcmp(options.covtype,'diag')
        for n=1:ndim
            regterm = [];
            if ~options.zeromean, regterm = prior.Mean.iS(n); end
            if ~isempty(orders)
                alphaterm = repmat( (metastates(k).alpha.Gam_shape ./  metastates(k).alpha.Gam_rate), ndim, 1);
                if ndim>1
                    regterm = [regterm; repmat(metastates(k).sigma.Gam_shape(:,n) ./ ...
                        metastates(k).sigma.Gam_rate(:,n), length(orders), 1).*alphaterm(:) ];
                else
                    regterm = [regterm; alphaterm(:)];
                end
            end
            if isempty(regterm), regterm = 0; end
            regterm = diag(regterm);
            prec =  regterm + Tfactor * (metastates(k).Omega.Gam_shape / metastates(k).Omega.Gam_rate(n)) * XXGXX;
            iSW(n,:,:) = prec;
            W(:,n) = prec \ ...
                ( ( Tfactor * (metastates(k).Omega.Gam_shape / metastates(k).Omega.Gam_rate(n)) * XX' .* ...
                repmat(Gamma(:,k)',size(XX,2),1) ) * X(:,n) );
        end
        
    elseif strcmp(options.covtype,'full')
        mlW = (Xdiv .* repmat(Gamma(:,k)',...
            (~options.zeromean)+ndim*length(orders),1) * X)';
        regterm = [];
        if ~options.zeromean, regterm = prior.Mean.iS; end
        if ~isempty(orders)
            sigmaterm = (metastates(k).sigma.Gam_shape ./ metastates(k).sigma.Gam_rate);
            sigmaterm = repmat(sigmaterm, length(orders), 1);
            alphaterm = repmat( (metastates(k).alpha.Gam_shape ./  metastates(k).alpha.Gam_rate), ndim^2, 1);
            alphaterm = alphaterm(:);
            regterm = [regterm; (alphaterm .* sigmaterm)];
        end
        if isempty(regterm), regterm = 0; end
        regterm = diag(regterm);
        prec = metastates(k).Omega.Gam_shape * metastates(k).Omega.Gam_irate;
        gram = kron(XXGXX, prec);
        iSW = regterm + Tfactor * gram;
        muW = iSW \ (Tfactor * gram) * mlW(:);
        W = reshape(muW,ndim,~options.zeromean+ndim*length(orders))';
    end
    metastates(k).W.Mu_W = (1-rho) * metastates(k).W.Mu_W + rho * W;
    metastates(k).W.iS_W = (1-rho) * metastates(k).W.iS_W + rho * iSW;
    metastates(k).W.S_W = inv(metastates(k).W.iS_W);
end
end


function metastates = updateBigSigma(metastates,rho,prior,options)
ndim = size(X,2); K = length(metastates);
for k=1:K
    shape = prior.sigma.Gam_shape + 0.5*length(options.orders);
    rate = prior.sigma.Gam_rate;
    for n1=1:ndim
        for n=1:n2
            index = n1 + (0:length(orders)-1)*ndim + ~options.zeromean;
            rate(n1,n2) = rate(n1,n2) + 0.5 * (metastates(k).W.Mu_W(index,n2)' * ...
                ((metastates(k).alpha.Gam_shape ./ metastates(k).alpha.Gam_rate') .* metastates(k).W.Mu_W(index,n2)) );
        end
    end
    if strcmp(options.covtype,'full') || strcmp(options.covtype,'uniquefull')
        for n1=1:ndim
            for n=1:n2
                index = (0:length(orders)-1) * ndim^2 + (n1-1) * ndim + n2 + (~options.zeromean)*ndim;
                rate(n1,n2) = rate(n1,n2) +  0.5 * sum((metastates(k).alpha.Gam_shape ./ metastates(k).alpha.Gam_rate') .* ...
                    diag(metastates(k).W.S_W(index,index) ));
            end
        end
    else
        for n1=1:ndim
            for n=1:n2
                index = n1 + (0:length(orders)-1)*ndim + ~options.zeromean;
                rate(n1,n2) = rate(n1,n2) +  0.5 * sum((metastates(k).alpha.Gam_shape ./ metastates(k).alpha.Gam_rate') .* ...
                        diag( permute(metastates(k).W.S_W(n2,index,index),[2 3 1]) )) ;
            end
        end
    end
    metastates(k).sigma.Gam_rate = (1-rho) * metastates(k).sigma.Gam_rate + rho * rate;
    metastates(k).sigma.Gam_shape = (1-rho) * metastates(k).sigma.Gam_shape + rho * shape;
end
end


function metastates = updateBigAlpha(metastates,rho,prior,options)
ndim = size(X,2); K = length(metastates);
for k=1:K,
    shape = prior.alpha.Gam_shape;
    rate = prior.alpha.Gam_rate;
    if ndim==1
        rate = rate + 0.5 * ( metastates(k).W.Mu_W(1+(~options.zeromean) : end).^2 )' + ...
            0.5 * diag(metastates(k).W.S_W(2:end,2:end))' ;
        shape = shape + 0.5;
    elseif strcmp(options.covtype,'full') || strcmp(options.covtype,'uniquefull')
        for n=1:ndim,
            for i=1:length(orders),
                index = (i-1)*ndim+n + ~options.zeromean;
                rate(i) = rate(i) + ...
                    0.5 * ( metastates(k).W.Mu_W(index,:) .* (metastates(k).sigma.Gam_shape(n,:) ./ metastates(k).sigma.Gam_rate(n,:) ) ) * ...
                    metastates(k).W.Mu_W(index,:)';
                index =  (i-1) * ndim^2 + (n-1) * ndim + (1:ndim) + (~options.zeromean)*ndim;  
                rate(i) = rate(i) + ...
                    0.5 * sum(diag(metastates(k).W.S_W(index,index))' .* (metastates(k).sigma.Gam_shape(n,:) ./ metastates(k).sigma.Gam_rate(n,:) ) );
            end
            shape = metastates(k).alpha.Gam_shape + 0.5 * ndim;
        end
    else
        for n=1:ndim,
            for i=1:length(orders),
                index = (i-1)*ndim+n + ~options.zeromean;
                rate(i) = rate(i) + ...
                    0.5 * ( (metastates(k).W.Mu_W(index,:) .* (metastates(k).sigma.Gam_shape(n,:) ./ metastates(k).sigma.Gam_rate(n,:)) ) * ...
                    metastates(k).W.Mu_W(index,:)' + sum( (metastates(k).sigma.Gam_shape(n,:) ./ metastates(k).sigma.Gam_rate(n,:)) .* ...
                    metastates(k).W.S_W(:,index,index)'));
            end;
            shape = shape + 0.5 * ndim;
        end
    end
    metastates(k).alpha.Gam_rate = (1-rho) * metastates(k).alpha.Gam_rate + rho * rate;
    metastates(k).alpha.Gam_shape = (1-rho) * metastates(k).alpha.Gam_shape + rho * shape;
end
end

function Y = trimX(X,T,order)
Y = zeros(sum(T)-length(T)*order,size(X,2));
for n = 1:length(T)
    tx = sum(T(1:n-1)) + (1+order:T(n));
    ty = sum(T(1:n-1)-(n-1)*order) + (1:T(n)-order);
    Y(ty,:) = X(tx,:);
end
end
