function [metahmm,info] = hmmsinit(Xin,T,options)
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
X = loadfile(Xin{1},T{1}); ndim = size(X,2);
subjfe_init = zeros(N,3);
loglik_init = zeros(N,1);
npred = length(options.orders)*ndim + (~options.zeromean);
info = struct();

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
for rep = 1:options.BIGinitrep
    
    % train individual HMMs
    I = randperm(N);
    for ii = 1:length(I)
        % read data
        i = I(ii);
        [X,XX,Y] = loadfile(Xin{i},T{i},options);
        if rep==1
            if ii==1
                range_data = range(X);
            else
                range_data = max(range_data,range(X));
            end
        end
        % Running the individual HMM
        options_copy = options; 
        options_copy = rmfield(options_copy,'BIGNbatch');
        options_copy = rmfield(options_copy,'orders');
        if length(T{i})==1, options_copy.useParallel = 0; end 
        [hmm_i,Gamma,Xi] = hmmmar(X,T{i},options_copy);
        if ii==1 % get priors
            Dir2d_alpha_prior = hmm_i.prior.Dir2d_alpha;
            Dir_alpha_prior = hmm_i.prior.Dir_alpha;
            hmm_i.train.orders = formorders(hmm_i.train.order,hmm_i.train.orderoffset,...
                hmm_i.train.timelag,hmm_i.train.exptimelag);
            Sind = formindexes(hmm_i.train.orders,hmm_i.train.S)==1;
            if ~hmm_i.train.zeromean, Sind = [true(1,ndim); Sind]; end
        end
        if options.BIGverbose
            fprintf('Init: repetition %d, subject %d \n',rep,ii);
        end
        if options.BIGuniqueTrans % update transition probabilities
            for trial=1:length(T{i})
                t = sum(T{i}(1:trial-1)) - options.order*(trial-1) + 1;
                Dir_alpha_init(:,i) = Dir_alpha_init(:,i) + Gamma(t,:)';
            end
            Dir2d_alpha_init(:,:,i) = squeeze(sum(Xi,1));
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
                    dist(j,k) = symm_kl_div(hmm_i.state(j), metahmm_init.state(k), Sind);
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
        if ii>1 && (strcmp(options.covtype,'uniquefull') || strcmp(options.covtype,'uniquediag'))
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
                    sum(subj_m_init(:,:,I(1:ii),k),3),options.covtype,Sind);
            elseif strcmp(options.covtype,'diag')
                metahmm_init.state(k) = metastate_new( ...
                    sum(subj_err_init(:,I(1:ii),k),2)' + hmm_i.state(k).prior.Omega.Gam_rate, ...
                    sum(subj_time_init(I(1:ii),k)) + hmm_i.state(k).prior.Omega.Gam_shape, ...
                    sum(subj_gram_init(:,:,I(1:ii),k),3) + 0.01 * eye(size(XX,2)), ...
                    sum(subj_m_init(:,:,I(1:ii),k),3),options.covtype,Sind);
            else
               metahmm_init.state(k) = metastate_new(metahmm_init.Omega.Gam_rate,...
                    metahmm_init.Omega.Gam_shape,...
                    sum(subj_gram_init(:,:,I(1:ii),k),3) + 0.01 * eye(size(XX,2)),...
                    sum(subj_m_init(:,:,I(1:ii),k),3),options.covtype,Sind);                
            end
        end
    end
        
    % adjust prior
    if rep==1
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
        metahmm_init.train.BIGNbatch = options.BIGNbatch;
        metahmm_init.train.Sind = Sind; 
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

    % update transition probabilities 
    if options.BIGuniqueTrans 
        metahmm_init.Dir_alpha = sum(Dir_alpha_init,2)' + Dir_alpha_prior;
        metahmm_init.Dir2d_alpha = sum(Dir2d_alpha_init,3) + Dir2d_alpha_prior;
        [metahmm_init.P,metahmm_init.Pi] = ...
            computePandPi(metahmm_init.Dir_alpha,metahmm_init.Dir2d_alpha);
    end
    
    % Compute free energy
    metahmm_init_i = metahmm_init;
    for i = 1:N
        [X,XX,Y] = loadfile(Xin{i},T{i},options);
        XX_i = cell(1); XX_i{1} = XX;
        data = struct('X',X,'C',NaN(sum(T{i})-length(T{i})*options.order,K));
        if ~options.BIGuniqueTrans
            metahmm_init_i = copyhmm(metahmm_init,...
                P_init(:,:,i),Pi_init(:,i)',Dir2d_alpha_init(:,:,i),Dir_alpha_init(:,i)');
        end
        [Gamma,~,Xi,l] = hsinference(data,T{i},metahmm_init_i,Y,[],XX_i);
        if options.BIGuniqueTrans
            subjfe_init(i,1:2) = evalfreeenergy([],T{i},Gamma,Xi,metahmm_init_i,[],[],[1 0 1 0 0]); % Gamma entropy&LL
        else
            subjfe_init(i,:) = evalfreeenergy([],T{i},Gamma,Xi,metahmm_init_i,[],[],[1 0 1 1 0]); 
        end
        loglik_init(i) = sum(l);
    end
    if options.BIGuniqueTrans
        subjfe_init(:,3) = evalfreeenergy([],[],[],[],metahmm_init,[],[],[0 0 0 1 0]) / N; % "share" P and Pi KL
    end
    statekl_init = sum(evalfreeenergy([],[],[],[],metahmm_init,[],[],[0 0 0 0 1])); % state KL
    fe = - sum(loglik_init) + sum(subjfe_init(:)) + statekl_init;
    
    if fe<best_fe
        best_fe = fe;
        metahmm = metahmm_init;
        info.P = P_init; info.Pi = Pi_init;
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

metahmm.prior.Dir_alpha_prior = Dir_alpha_prior;
metahmm.prior.Dir2d_alpha_prior = Dir2d_alpha_prior;

end


