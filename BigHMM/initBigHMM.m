
subjfe_init = zeros(N,3);
loglik_init = zeros(N,1);
npred = length(options.orders)*ndim + (~options.zeromean);

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
        [X,XX,Y] = loadfile(Xin{i},T{i},options);
        if cycle==1
            if ii==1
                range_data = range(X);
            else
                range_data = max(range_data,range(X));
            end
        end
        % Running the individual HMM
        [hmm_i,Gamma,Xi] = hmmmar(X,T{i},options);
        if ii==1 % get priors
            Dir2d_alpha_prior = hmm_i.prior.Dir2d_alpha;
            Dir_alpha_prior = hmm_i.prior.Dir_alpha;
        end
        if BIGverbose
            fprintf('Init run %d, subject %d \n',cycle,ii);
        end
        if BIGuniqueTrans % update transition probabilities
            Dir_alpha_init(:,i) = 0;
            for trial=1:length(T{i})
                t = sum(T{i}(1:trial-1)) - options.order*(trial-1) + 1;
                Dir_alpha_init(:,i) = Dir_alpha_init(:,i) + Gamma(t,:)';
            end
            Dir2d_alpha_init(:,:,i) = squeeze(sum(Xi,1));
            % one P/Pi for all subjects
            Dir2d_alpha_init_pre = sum(Dir2d_alpha_init(:,:,I(1:ii)),3) + Dir2d_alpha_prior;
            Dir_alpha_init_pre = sum(Dir_alpha_init(:,I(1:ii)),2)' + Dir_alpha_prior;
            [P_init,Pi_init] = computePandPi(Dir_alpha_init_pre,Dir2d_alpha_init_pre);
        else
            P_init(:,:,i) = hmm_i.P; Pi_init(:,i) = hmm_i.Pi';
            Dir2d_alpha_init(:,:,i) = hmm_i.Dir2d_alpha;
            Dir_alpha_init(:,i) = hmm_i.Dir_alpha';
        end
        K_i = length(hmm_i.state);
        % Reassigning ordering of the states according the closest metahmm
        if ii==1
            assig = 1:K_i;
            if K_i<K,
                warning('The first HMM run needs to return K states, you might want to start again..\n')
            end
            metahmm_init = struct('train',hmm_i.train);
            if strcmp(options.covtype,'uniquefull') || strcmp(options.covtype,'uniquediag')
                metahmm_init.Omega = hmm_i.Omega; 
            end
        else
            dist = Inf(K_i,K);
            for j = 1:K_i
                for k = 1:K
                    dist(j,k) = symm_kl_div(hmm_i.state(j), metahmm_init.state(k));
                end
            end
            assig = munkres(dist); % linear assignment problem
        end
        % update sufficient statistics
        for k=1:K_i,
            XG = XX' .* repmat(Gamma(:,k)',size(XX,2),1);
            subj_m_init(:,:,i,assig(k)) = XG * Y;
            subj_gram_init(:,:,i,assig(k)) = XG * XX;
            if strcmp(options.covtype,'full')
                subj_err_init(:,:,i,assig(k)) = hmm_i.state(k).Omega.Gam_rate - ...
                    hmm_i.state(k).prior.Omega.Gam_rate;
                subj_time_init(i,assig(k)) = hmm_i.state(k).Omega.Gam_shape - ...
                    hmm_i.state(k).prior.Omega.Gam_shape;
            elseif strcmp(options.covtype,'diag')
                subj_err_init(:,i,assig(k)) = hmm_i.state(k).Omega.Gam_rate' - ...
                    hmm_i.state(k).prior.Omega.Gam_rate';
                subj_time_init(i,assig(k)) = hmm_i.state(k).Omega.Gam_shape - ...
                    hmm_i.state(k).prior.Omega.Gam_shape;
            end
            % cov mats: note also that these are the individual ones, and,
            % hence, an underestimation of the group ones
        end
        if strcmp(options.covtype,'uniquefull') || strcmp(options.covtype,'uniquediag')
            metahmm_init.Omega.Gam_shape = metahmm_init.Omega.Gam_shape + ...
                hmm_i.Omega.Gam_shape - hmm_i.prior.Omega.Gam_shape;
            metahmm_init.Omega.Gam_rate = metahmm_init.Omega.Gam_rate + ...
                hmm_i.Omega.Gam_rate - hmm_i.prior.Omega.Gam_rate;
        end
        % updating the metahmm
        for k = 1:K_i
            if strcmp(options.covtype,'full')
                metahmm_init.state(k) = metastate_new( ...
                    sum(subj_err_init(:,:,I(1:ii),k),3) + hmm_i.state(k).prior.Omega.Gam_rate, ...
                    sum(subj_time_init(I(1:ii),k)) + hmm_i.state(k).prior.Omega.Gam_shape, ...
                    sum(subj_gram_init(:,:,I(1:ii),k),3) + 0.01 * eye(size(XX,2)), ...
                    sum(subj_m_init(:,:,I(1:ii),k),3));
            elseif strcmp(options.covtype,'diag')
                metahmm_init.state(k) = metastate_new( ...
                    sum(subj_err_init(:,I(1:ii),k),2)' + hmm_i.state(k).prior.Omega.Gam_rate, ...
                    sum(subj_time_init(I(1:ii),k)) + hmm_i.state(k).prior.Omega.Gam_shape, ...
                    sum(subj_gram_init(:,:,I(1:ii),k),3) + 0.01 * eye(size(XX,2)), ...
                    sum(subj_m_init(:,:,I(1:ii),k),3));
            else
               metahmm_init.state(k) = metastate_new([],[], ...
                    sum(subj_gram_init(:,:,I(1:ii),k),3) + 0.01 * eye(size(XX,2)),...
                    sum(subj_m_init(:,:,I(1:ii),k),3));                
            end
        end
        metahmm_init = adjSw_in_metastate(metahmm_init); % adjust dimension of S_W
    end
        
    % adjust prior
    if cycle==1
        if isempty(options.BIGprior)
            for k = 1:K
                metahmm_init.state(k).prior = hmm_i.state(k).prior;
                if isfield(metahmm_init.state(k).prior,'Omega')
                    if strcmp(options.covtype,'diag')
                        metahmm_init.state(k).prior.Omega.Gam_rate = 0.5 * range_data;
                    elseif strcmp(options.covtype,'full')
                        metahmm_init.state(k).prior.Omega.Gam_rate = diag(range_data);
                    end
                end
                if isfield(metahmm_init.state(k).prior,'Mean')
                    metahmm_init.state(k).prior.Mean.Mu = zeros(ndim,1);
                    metahmm_init.state(k).prior.Mean.S = ((range_data/2).^2)';
                    metahmm_init.state(k).prior.Mean.iS = 1 ./ metahmm_init.state(k).prior.Mean.S;
                end
            end
            metahmm_init.prior = hmm_i.prior;
            if strcmp(options.covtype,'uniquediag')
                metahmm_init.prior.Omega.Gam_rate = 0.5 * range_data;
            elseif strcmp(options.covtype,'uniquefull')
                metahmm_init.prior.Omega.Gam_rate = diag(range_data);
            end
        else
            for k = 1:K
                metahmm_init.state(k).prior = options.BIGprior.state(k).prior;
            end
            metahmm_init.prior.Dir2d_alpha = options.BIGprior.Dir2d_alpha;
            metahmm_init.prior.Dir_alpha = options.BIGprior.Dir_alpha;
        end
        metahmm_init.K = K;
    end
    
    % distribution of sigma and alpha, variances of the MAR coeff distributions
    if ~isempty(options.orders)
        for k=1:K,
            metahmm_init.state(k).alpha.Gam_shape = metahmm_init.state(k).prior.alpha.Gam_shape;
            metahmm_init.state(k).alpha.Gam_rate = metahmm_init.state(k).prior.alpha.Gam_rate;
        end
        metahmm_init = updateSigma(metahmm_init);
        metahmm_init = updateAlpha(metahmm_init);
    end

    % Compute Gamma to get an
    % unbiased group estimation of the metastate covariance matrices;
    % obtaining subject parameters (transition probs)  
    if BIGuniqueTrans
        Dir2d_alpha_init_pre = sum(Dir2d_alpha_init,3) + Dir2d_alpha_prior;
        Dir_alpha_init_pre = sum(Dir_alpha_init,2)' + Dir_alpha_prior;
        [P_init_pre,Pi_init_pre] = computePandPi(Dir_alpha_init_pre,Dir2d_alpha_init_pre);
    end
    for i = 1:N
        [X,XX,Y] = loadfile(Xin{i},T{i},options);
        XX_i = cell(1); XX_i{1} = XX;
        data = struct('X',X,'C',NaN(sum(T{i})-length(T{i})*options.order,K));
        if BIGuniqueTrans
            metahmm_init_i = copyhmm(metahmm_init,...
                P_init_pre,Pi_init_pre,Dir2d_alpha_init_pre,Dir_alpha_init_pre);
        else
            metahmm_init_i = copyhmm(metahmm_init,...
                P_init(:,:,i),Pi_init(:,i)',Dir2d_alpha_init(:,:,i),Dir_alpha_init(:,i)');
        end
        [Gamma,~,Xi,l] = hsinference(data,T{i},metahmm_init_i,Y,[],XX_i);
        metahmm_init_i = hsupdate(Xi,Gamma,T{i},metahmm_init_i);
        if BIGuniqueTrans
            for trial=1:length(T{i})
                t = sum(T{i}(1:trial-1)) - options.order*(trial-1) + 1;
                Dir_alpha_init(:,i) = Dir_alpha_init(:,i) + Gamma(t,:)';
            end
            Dir2d_alpha_init(:,:,i) = squeeze(sum(Xi,1));
        else
            P_init(:,:,i) = metahmm_init_i.P; Pi_init(:,i) = metahmm_init_i.Pi'; % one per subject, not like pure group HMM
            Dir2d_alpha_init(:,:,i) = metahmm_init_i.Dir2d_alpha; Dir_alpha_init(:,i) = metahmm_init_i.Dir_alpha';
        end
        if strcmp(options.covtype,'uniquefull'), EE = zeros(ndim);
        elseif strcmp(options.covtype,'uniquediag'), EE = zeros(1,ndim);
        end
        for k=1:K
            if ~isempty(options.orders) || (~options.zeromean)
                E = Y - XX * metahmm_init.state(k).W.Mu_W; % using the current mean estimation
            else
                E = X;
            end
            if strcmp(options.covtype,'full')
                subj_err_init(:,:,i,k) = ((E' .* repmat(Gamma(:,k)',size(E,2),1)) * E);
                metahmm_init.state(k).Omega.Gam_rate = sum(subj_err_init(:,:,I(1:ii),k),3) + ...
                    hmm_i.state(k).prior.Omega.Gam_rate;
                subj_time_init(i,k) = sum(Gamma(:,k));
                metahmm_init.state(k).Omega.Gam_shape = sum(subj_time_init(I(1:ii),k)) + ...
                    hmm_i.state(k).prior.Omega.Gam_shape;
            elseif strcmp(options.covtype,'diag')
                subj_err_init(:,i,k) = ( sum( (E.^2) .* repmat(Gamma(:,k),1,size(E,2)) ) )';
                metahmm_init.state(k).Omega.Gam_rate = sum(subj_err_init(:,I(1:ii),k),2)' + ...
                    hmm_i.state(k).prior.Omega.Gam_rate;
                subj_time_init(i,k) = sum(Gamma(:,k));
                metahmm_init.state(k).Omega.Gam_shape = sum(subj_time_init(I(1:ii),k)) + ...
                    hmm_i.state(k).prior.Omega.Gam_shape;
            elseif strcmp(options.covtype,'uniquefull')
                EE = EE + ((E' .* repmat(Gamma(:,k)',size(E,2),1)) * E);
            elseif strcmp(options.covtype,'uniquediag')
                EE = EE + ( sum( (E.^2) .* repmat(Gamma(:,k),1,size(E,2)) ) );
            end
        end
        if strcmp(options.covtype,'uniquefull') || strcmp(options.covtype,'uniquediag') 
            metahmm_init.Omega.Gam_rate = EE + metahmm_init.prior.Omega.Gam_rate;
        end
        if BIGuniqueTrans
            subjfe_init(i,1:2) = evalfreeenergy([],T{i},Gamma,Xi,metahmm_init_i,[],[],[1 0 1 0 0]); % Gamma entropy&LL
        else
            subjfe_init(i,:) = evalfreeenergy([],T{i},Gamma,Xi,metahmm_init_i,[],[],[1 0 1 1 0]); 
        end
        loglik_init(i) = sum(l);
    end
    if BIGuniqueTrans
        metahmm_init.Dir_alpha = sum(Dir_alpha_init,2)' + Dir_alpha_prior;
        metahmm_init.Dir2d_alpha = sum(Dir2d_alpha_init,3) + Dir2d_alpha_prior;
        [metahmm_init.P_init,metahmm_init.Pi_init] = ...
            computePandPi(metahmm_init.Dir_alpha,metahmm_init.Dir2d_alpha);
        subjfe_init(:,3) = evalfreeenergy([],[],[],[],metahmm_init,[],[],[0 0 0 1 0]) / N; % "share" KL
    end
    statekl_init = sum(evalfreeenergy([],[],[],[],metahmm_init,[],[],[0 0 0 0 1])); % state KL
    fe = - sum(loglik_init) + sum(subjfe_init(:)) + statekl_init;
    
    if fe<best_fe
        best_fe = fe;
        metahmm = metahmm_init;
        P = P_init; Pi = Pi_init;
        Dir2d_alpha = Dir2d_alpha_init; Dir_alpha = Dir_alpha_init;
        subjfe = subjfe_init;
        loglik = loglik_init;
        statekl = statekl_init;
        fehist = (-sum(loglik) + sum(statekl) + sum(sum(subjfe(:,:))));
    end
    
    if BIGverbose
        fprintf('Init run %d, free energy = %g (best=%g) \n',cycle,fe,best_fe);
    end
    
end

metahmm.prior.Dir_alpha_prior = Dir_alpha_prior;
metahmm.prior.Dir2d_alpha_prior = Dir2d_alpha_prior;


%if BIGverbose
%    fprintf('Cycle 1, free energy: %g \n',fehist);
%end

% else % initial metahmm specified by the user
%     
%     metahmm = options.metahmm; options = rmfield(options,'metahmm');
%     if strcmp(options.covtype,'diag')
%         gramm = [];
%     elseif strcmp(options.covtype,'full')
%         gramm = [];
%     elseif strcmp(options.covtype,'uniquediag')
%         gramm = zeros(1,ndim);  
%     else % uniquefull
%         gramm = zeros(ndim,ndim);  
%     end
%        
%     P = zeros(K,K,N); Pi = zeros(K,N);
%     Dir2d_alpha = zeros(K,K,N); Dir_alpha = zeros(K,N);
% 
%     % collect some stats
%     for i = 1:N
%         X = loadfile(Xin{i});
%         if i==1
%             options.inittype = 'random';
%             options.cyc = 1;
%             hmm0 = hmmmar(X,T{i},options);
%             range_data = range(X);
%         else
%             range_data = max(range_data,range(X));
%         end
%         if strcmp(options.covtype,'uniquefull')
%             gramm = gramm + X' * X;
%         elseif strcmp(options.covtype,'uniquediag')
%             gramm = gramm + sum(X.^2);
%         end
%     end
%     
%     % set prior
%     prior = struct();
%     prior.Omega = struct();
%     if strcmp(options.covtype,'uniquediag') || strcmp(options.covtype,'diag')
%         prior.Omega.Gam_shape = 0.5 * (ndim+0.1-1);
%         prior.Omega.Gam_rate = 0.5 * range_data;
%     else
%         prior.Omega.Gam_shape = ndim+0.1-1;
%         prior.Omega.Gam_rate = diag(range_data);
%     end
%     prior.Mean = struct();
%     prior.Mean.S = (range_data/2).^2;
%     prior.Mean.iS = 1 ./ prior.Mean.S;
%     prior.Mean.Mu = zeros(1,ndim);
%     if ~isempty(options.orders)
%         prior.sigma = struct();
%         prior.sigma.Gam_shape = 0.1*ones(ndim,ndim); %+ 0.05*eye(ndim);
%         prior.sigma.Gam_rate = 0.1*ones(ndim,ndim);%  + 0.05*eye(ndim);
%         prior.alpha = struct();
%         prior.alpha.Gam_shape = 0.1;
%         prior.alpha.Gam_rate = 0.1*ones(1,length(orders));
%     end
%     
%     % Init subject models and free energy computation
%     for i = 1:N
%         X = loadfile(Xin{i});
%         data = struct('X',X,'C',NaN(size(X,1),K));
%         hmm = loadhmm(hmm0,T{i},K,metahmm,[],[],[],[],gramm,prior);
%         % get gamma
%         [Gamma,~,Xi,l] = hsinference(data,T{i},hmm,[]);
%         % compute transition prob
%         hmm = hsupdate(Xi,Gamma,T{i},hmm);
%         if BIGuniqueTrans
%             for trial=1:length(T{i})
%                 t = sum(T{i}(1:trial-1)) - options.order*(trial-1) + 1;
%                 Dir_alpha(:,i) = Dir_alpha(:,i) + Gamma(t,:)';
%             end
%             Dir2d_alpha(:,:,i) = squeeze(sum(Xi,1));
%         else
%             P(:,:,i) = hmm.P; Pi(:,i) = hmm.Pi'; % one per subject, not like pure group HMM
%             Dir2d_alpha(:,:,i) = hmm.Dir2d_alpha; Dir_alpha(:,i) = hmm.Dir_alpha';
%         end
%         % compute free energy
%         loglik(i,1) = sum(l);  
%         subjfe(i,1,1) = - GammaEntropy(Gamma,Xi,T{i},0); 
%         subjfe(i,2,1) = - GammaavLL(hmm,Gamma,Xi,T{i});
%         if ~BIGuniqueTrans
%             subjfe(i,3,1) = + KLtransition(hmm);
%         end
%     end
%     if BIGuniqueTrans
%         hmm.Dir_alpha = sum(Dir_alpha,2)' + Dir_alpha_prior;
%         hmm.Dir2d_alpha = sum(Dir2d_alpha,3) + Dir2d_alpha_prior;
%         [P,Pi] = computePandPi(hmm.Dir_alpha,hmm.Dir2d_alpha);
%         subjfe(:,3,1) = KLtransition(hmm) / N; % "share" the common KL
%     end
%     for k = 1:K
%         statekl(k,1) = KLstate(metahmm(k),prior,options.covtype,options.zeromean);
%     end
%     fehist = sum(- loglik(:,1) + sum(sum(subjfe(:,:,1))) + sum(statekl(:,1)));
%     
%     if BIGverbose
%         fprintf('Cycle 1, free energy: %g \n',fehist);
%     end
%   
% end

clear metastate_gamma_init metastate_m_init metastate_gram_init metahmm_init metahmm_init_i
clear gram_init subjfe_init loglik_init statekl_init Pi_init P_init Dir2d_alpha_init Dir_alpha_init
