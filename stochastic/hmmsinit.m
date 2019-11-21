function [hmm,info] = hmmsinit(Xin,T,options)
% Initialisation before stochastic HMM variational inference
%
% INPUTS
% Xin: cell with strings referring to the files containing each subject's data, 
%       or cell with with matrices (time points x channels) with each
%       subject's data
% T: cell of vectors, where each element has the length of each trial per
%       subject. Dimension of T{n} has to be (1 x nTrials)
% options: HMM options for both the subject and the group runs
%
% Diego Vidaurre, OHBA, University of Oxford (2016)

N = length(T); K = options.K;
X = loadfile(Xin{1},T{1},options); ndim = size(X,2);
subjfe_init = zeros(N,3);
loglik_init = zeros(N,1);
pcaprec = options.pcapred>0;

S = options.S==1; regressed = sum(S,1)>0;
if isfield(options,'B') && ~isempty(options.B)
    npred = length(options.orders)*size(options.B,2) + (~options.zeromean);
elseif pcaprec
    npred = options.pcapred + (~options.zeromean);
else
    npred = length(options.orders)*ndim + (~options.zeromean);
end
info = struct();
if isfield(options,'initial_hmm') 
    initial_hmm = options.initial_hmm; 
    options = rmfield(options,'initial_hmm');
else 
    initial_hmm = [];
end
tp_less = max(options.embeddedlags) + max(-options.embeddedlags);
if options.downsample > 0
    downs_ratio = (options.downsample/options.Fs);
else
    downs_ratio = 1; 
end

% init sufficient statistics
subj_m_init = zeros(npred,ndim,N,K);
subj_gram_init = zeros(npred,npred,N,K);
if strcmp(options.covtype,'diag')
    subj_err_init = zeros(ndim,N,K); 
elseif strcmp(options.covtype,'full')
    subj_err_init = zeros(ndim,ndim,N,K);  
else
    subj_err_init = [];
end
subj_time_init = zeros(N,K);

% init subject parameters
Dir2d_alpha_init = zeros(K,K,N); Dir_alpha_init = zeros(K,N);

best_fe = Inf;
best_maxFO = 1;

for rep = 1:options.BIGinitrep
    
    % train individual HMMs
    if ~isempty(initial_hmm)  
        II = randperm(length(initial_hmm));
        I = cell(1,length(II));
        for ii = 1:length(I), I{ii} = initial_hmm{II(ii)}.subset; end
    else
        I = randpermNK(N,round(N/options.BIGNinitbatch));
    end
    covered = [];
    for ii = 1:length(I)
        % read data
        subset = I{ii};
        if size(subset,1)==1, subset = subset'; end
        covered = [covered; subset];
        [X,XX,Y,Ti] = loadfile(Xin(subset),T(subset),options);
        if rep==1
            if ii==1
                range_data = range(Y);
            else
                range_data = max(range_data,range(Y));
            end
        end
        % Running the individual HMM
        if ~isempty(initial_hmm)
            hmm_i = versCompatibilityFix(initial_hmm{II(ii)});
            [Gamma,~,Xi] = hsinference(X,Ti,hmm_i,Y,options,XX);
            for i=1:length(subset)
                tt = (1:Ti(i)-hmm_i.train.order) + sum(Ti(1:i-1)) - (i-1)*hmm_i.train.order;
                checkGamma(Gamma(tt,:),Ti(i),hmm_i.train,subset(i));
            end
        else
            options_copy = options;
            removesoptions;
            options = rmfield(options,'orders');
            options.pca = 0; % done in loadfile.m
            options.pca_spatial = 0;
            options.embeddedlags = 0; % done in loadfile.m
            options.filter = [];
            options.detrend = 0; 
            options.onpower = 0; 
            options.downsample = 0; % done in loadfile.m
            options.grouping = [];
            if isfield(options,'A')
                options = rmfield(options,'A');
            end
            %options.initcriterion = 'FreeEnergy';
            if length(Ti)==1, options.useParallel = 0; end
            [hmm_i,Gamma,Xi] = hmmmar(X,Ti,options);
            options = options_copy;
            hmm_i.train.pca = options.pca; 
            hmm_i.train.pca_spatial = options.pca_spatial;
            hmm_i.train.embeddedlags = options.embeddedlags;
            hmm_i.train.filter = options.filter;
            hmm_i.train.detrend = options.detrend;
            hmm_i.train.onpower = options.onpower;
            hmm_i.train.downsample = options.downsample;
            hmm_i.train.useParallel = options.useParallel;
            hmm_i.train.grouping = options.grouping;
        end
        if ii==1 % get priors
            Dir2d_alpha_prior = hmm_i.prior.Dir2d_alpha;
            Dir_alpha_prior = hmm_i.prior.Dir_alpha;
            hmm_i.train.orders = formorders(hmm_i.train.order,hmm_i.train.orderoffset,...
                hmm_i.train.timelag,hmm_i.train.exptimelag);
            if options.pcapred>0
                Sind = true(options.pcapred,ndim);
            else
                Sind = formindexes(hmm_i.train.orders,hmm_i.train.S)==1;
            end
            if ~hmm_i.train.zeromean, Sind = [true(1,ndim); Sind]; end
        end
        if options.BIGverbose
            fprintf('Init: repetition %d, batch %d \n',rep,ii);
        end
        % update transition probabilities
        tacc = 0; tacc2 = 0;
        for i=1:length(subset)
            subj = subset(i);
            Tsubj = ceil(downs_ratio*(T{subj}-tp_less)); 
            for trial=1:length(Tsubj)
                t = tacc + 1;
                Dir_alpha_init(:,subj) = Dir_alpha_init(:,subj) + Gamma(t,:)';
                tacc = tacc + Tsubj(trial) - hmm_i.train.maxorder;
                t = tacc2 + (1:Tsubj(trial)-hmm_i.train.maxorder-1);
                Dir2d_alpha_init(:,:,subj) = Dir2d_alpha_init(:,:,subj) + squeeze(sum(Xi(t,:,:),1));
                tacc2 = tacc2 + Tsubj(trial) - hmm_i.train.maxorder - 1;
            end
        end
        K_i = length(hmm_i.state);
        % Reassigning ordering of the states according the closest hmm
        if ii==1
            assig = 1:K_i;
            if K_i<K
                warning('The first HMM run needs to return K states, you might want to start again..\n')
            end
            hmm_init = struct('train',hmm_i.train);
            hmm_init.train.active = ones(1,K);
            if strcmp(options.covtype,'uniquefull') || strcmp(options.covtype,'uniquediag')
                hmm_init.Omega = hmm_i.Omega; 
            end
        else
            dist = Inf(K_i,K);
            for j = 1:K_i
                for k = 1:K
                    dist(j,k) = symm_kl_div(hmm_i.state(j), hmm_init.state(k), Sind, hmm_i.train.covtype);
                end
            end
            assig = munkres(dist); % linear assignment problem
        end
        % update sufficient statistics
        tacc = 0; 
        for i = 1:length(subset)
            subj = subset(i);
            Tsubj = ceil(downs_ratio*(T{subj}-tp_less)); 
            t = tacc + (1 : (sum(Tsubj)-length(Tsubj)*hmm_i.train.maxorder));
            tacc = tacc + length(t);
            for k=1:K_i
                XG = XX(t,:)' .* repmat(Gamma(t,k)',npred,1);
                subj_m_init(:,:,subj,assig(k)) = XG * Y(t,:);
                subj_gram_init(:,:,subj,assig(k)) = XG * XX(t,:);
                if strcmp(options.covtype,'full')
                    subj_err_init(:,:,subj,assig(k)) = hmm_i.state(k).Omega.Gam_rate - ...
                        hmm_i.state(k).prior.Omega.Gam_rate;
                    subj_time_init(subj,assig(k)) = hmm_i.state(k).Omega.Gam_shape - ...
                        hmm_i.state(k).prior.Omega.Gam_shape;
                elseif strcmp(options.covtype,'diag')
                    subj_err_init(regressed,subj,assig(k)) = hmm_i.state(k).Omega.Gam_rate(regressed)' - ...
                        hmm_i.state(k).prior.Omega.Gam_rate(regressed)';
                    subj_time_init(subj,assig(k)) = hmm_i.state(k).Omega.Gam_shape - ...
                        hmm_i.state(k).prior.Omega.Gam_shape;
                end
                % cov mats: note also that these are the individual ones, and,
                % hence, an underestimation of the group ones
            end
        end
        if ii>1 && (strcmp(options.covtype,'uniquefull') || strcmp(options.covtype,'uniquediag'))
            hmm_init.Omega.Gam_shape = hmm_init.Omega.Gam_shape + ...
                hmm_i.Omega.Gam_shape - hmm_i.prior.Omega.Gam_shape;
            hmm_init.Omega.Gam_rate = hmm_init.Omega.Gam_rate + ...
                hmm_i.Omega.Gam_rate - hmm_i.prior.Omega.Gam_rate;
            if strcmp(options.covtype,'uniquediag')
                hmm_init.Omega.Gam_rate(~regressed) = 0;
            end
        end
        % updating the hmm
        for k = 1:K_i
            if strcmp(options.covtype,'full')
                hmm_init.state(k) = state_snew( ...
                    sum(subj_err_init(:,:,covered,k),3) + hmm_i.state(k).prior.Omega.Gam_rate, ...
                    sum(subj_time_init(covered,k)) + hmm_i.state(k).prior.Omega.Gam_shape, ...
                    sum(subj_gram_init(:,:,covered,k),3) + 0.01 * eye(npred), ...
                    sum(subj_m_init(:,:,covered,k),3),options.covtype,Sind);
            elseif strcmp(options.covtype,'diag')
                hmm_init.state(k) = state_snew( ...
                    sum(subj_err_init(:,covered,k),2)' + hmm_i.state(k).prior.Omega.Gam_rate, ...
                    sum(subj_time_init(covered,k)) + hmm_i.state(k).prior.Omega.Gam_shape, ...
                    sum(subj_gram_init(:,:,covered,k),3) + 0.01 * eye(npred), ...
                    sum(subj_m_init(:,:,covered,k),3),options.covtype,Sind);
            else
               hmm_init.state(k) = state_snew(hmm_init.Omega.Gam_rate,...
                    hmm_init.Omega.Gam_shape,...
                    sum(subj_gram_init(:,:,covered,k),3) + 0.01 * eye(npred),...
                    sum(subj_m_init(:,:,covered,k),3),options.covtype,Sind);                
            end
        end
    end
    
    % adjust prior
    if rep==1
        if isempty(options.BIGprior)
            for k = 1:K
                hmm_init.state(k).prior = hmm_i.state(k).prior;
                if isfield(hmm_init.state(k).prior,'Omega')
                    if strcmp(options.covtype,'diag')
                        hmm_init.state(k).prior.Omega.Gam_rate = 0.5 * range_data;
                    elseif strcmp(options.covtype,'full')
                        hmm_init.state(k).prior.Omega.Gam_rate = diag(range_data);
                    end
                end
                if isfield(hmm_init.state(k).prior,'Mean')
                    hmm_init.state(k).prior.Mean.Mu = zeros(ndim,1);
                    hmm_init.state(k).prior.Mean.S = ((range_data/2).^2)';
                    hmm_init.state(k).prior.Mean.iS = 1 ./ hmm_init.state(k).prior.Mean.S;
                end
            end
            hmm_init.prior = hmm_i.prior;
            if strcmp(options.covtype,'uniquediag')
                hmm_init.prior.Omega.Gam_rate = 0.5 * range_data;
            elseif strcmp(options.covtype,'uniquefull')
                hmm_init.prior.Omega.Gam_rate = diag(range_data);
            end
        else
            for k = 1:K
                hmm_init.state(k).prior = options.BIGprior.state(k).prior;
            end
            hmm_init.prior.Dir2d_alpha = options.BIGprior.Dir2d_alpha;
            hmm_init.prior.Dir_alpha = options.BIGprior.Dir_alpha;
        end
        hmm_init.K = K;
        hmm_init.train.BIGNbatch = options.BIGNbatch;
        hmm_init.train.Sind = Sind; 
    end
    
    % distribution of sigma and alpha, variances of the MAR coeff distributions
    if ~isempty(options.orders)
        if pcaprec
            hmm_init = updateBeta(hmm_init);
        else
            for k=1:K
                hmm_init.state(k).alpha.Gam_shape = hmm_init.state(k).prior.alpha.Gam_shape;
                hmm_init.state(k).alpha.Gam_rate = hmm_init.state(k).prior.alpha.Gam_rate;
            end
            hmm_init = updateSigma(hmm_init);
            hmm_init = updateAlpha(hmm_init);
        end
    end

    % update transition probabilities
    hmm_init.Dir_alpha = sum(Dir_alpha_init,2)' + Dir_alpha_prior;
    hmm_init.Dir2d_alpha = sum(Dir2d_alpha_init,3) + Dir2d_alpha_prior;
    [hmm_init.P,hmm_init.Pi] =  computePandPi(hmm_init.Dir_alpha,hmm_init.Dir2d_alpha);
    
    % Compute free energy and maxFO
    hmm_init_i = hmm_init;
    maxFO = zeros(N,1);
    for i = 1:N
        [X,XX,Y,Ti] = loadfile(Xin{i},T{i},options);
        data = struct('X',X,'C',NaN(size(XX,1),K));
        [Gamma,~,Xi,l] = hsinference(data,Ti,hmm_init_i,Y,[],XX);
        checkGamma(Gamma,Ti,hmm_init_i.train,i);
        subjfe_init(i,1:2) = evalfreeenergy([],Ti,Gamma,Xi,hmm_init_i,[],[],[1 0 1 0 0]); % Gamma entropy&LL
        loglik_init(i) = sum(l);
        maxFO(i) = mean(getMaxFractionalOccupancy(Gamma,Ti,options));
    end
    subjfe_init(:,3) = evalfreeenergy([],[],[],[],hmm_init,[],[],[0 0 0 1 0]) / N; % "share" P and Pi KL
    statekl_init = sum(evalfreeenergy([],[],[],[],hmm_init,[],[],[0 0 0 0 1])); % state KL
    fe = - sum(loglik_init) + sum(subjfe_init(:)) + statekl_init;
    maxFO = mean(maxFO);
    
    this_is_best = false; 
    if isfield(options,'initcriterion') && strcmpi(options.initcriterion,'FreeEnergy')
        if fe < best_fe
            best_fe = fe;
            this_is_best = true;
        end
    else
        if maxFO < best_maxFO
            best_maxFO = maxFO;
            this_is_best = true;
        end
    end
    if this_is_best
        hmm = hmm_init;
        info.Dir2d_alpha = Dir2d_alpha_init; info.Dir_alpha = Dir_alpha_init;
        info.subjfe = subjfe_init;
        info.loglik = loglik_init;
        info.statekl = statekl_init;
        info.fehist = (-sum(info.loglik) + sum(info.statekl) + sum(sum(info.subjfe)));
    end

    if options.BIGverbose
        fprintf('Init run %d, free energy = %g (best=%g) \n',rep,fe,best_fe);
    end
    
end

hmm.prior.Dir_alpha_prior = Dir_alpha_prior;
hmm.prior.Dir2d_alpha_prior = Dir2d_alpha_prior;


end
