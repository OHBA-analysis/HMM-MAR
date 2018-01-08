function [hmm,info] = hmmsinitg(Xin,T,options,GammaInit)
% Initialisation before stochastic HMM variational inference, when Gamma
% is provided
%
% INPUTS
% Xin: cell with strings referring to the files containing each subject's data,
%       or cell with with matrices (time points x channels) with each
%       subject's data
% T: cell of vectors, where each element has the length of each trial per
%       subject. Dimension of T{n} has to be (1 x nTrials)
% options: HMM options for both the subject and the group runs
% GammaInit: the initial state time courses
%
% Diego Vidaurre, OHBA, University of Oxford (2016)

N = length(T); K = size(GammaInit,2);
X = loadfile(Xin{1},T{1},options); ndim = size(X,2);
subjfe_init = zeros(N,3);
loglik_init = zeros(N,1);
pcaprec = options.pcapred>0;
S = options.S==1; regressed = sum(S,1)>0;
if pcaprec
    npred = options.pcapred + (~options.zeromean);
else
    if isfield(options,'B') && ~isempty(options.B)
        Q = size(options.B,2);
    else
        Q = ndim;
    end
    npred = length(options.orders)*Q + (~options.zeromean);
end
info = struct();

% init sufficient statistics
subj_m_init = zeros(npred,ndim,K);
subj_gram_init = zeros(npred,npred,K);
if strcmp(options.covtype,'diag')
    subj_err_init = zeros(1,ndim,K); subj_time_init = zeros(K,1);
elseif strcmp(options.covtype,'full')
    subj_err_init = zeros(ndim,ndim,K); subj_time_init = zeros(K,1);
elseif strcmp(options.covtype,'uniquediag')
    subj_err_init = zeros(1,ndim); 
else
    subj_err_init = zeros(ndim,ndim);  
end

% init subject parameters
Dir2d_alpha_init = zeros(K,K,N); Dir_alpha_init = zeros(K,N);

% Collect sufficient statistics for W
tsum = 0;
for i = 1:N
    % read data
    [X,XX,Y,Ti] = loadfile(Xin{i},T{i},options);
    % check options
    if i==1
        useParallel = length(T)>1; 
        if isfield(options,'useParallel'), useParallel = options.useParallel; end
        options.useParallel = 0;
        options = checkoptions(options,X,Ti,0);
        if options.pcapred>0
            Sind = true(options.pcapred,ndim);
        else
            Sind = formindexes(options.orders,options.S)==1;
        end        
        if ~options.zeromean, Sind = [true(1,ndim); Sind]; end
        options.useParallel = useParallel;
    end
    % get the range of the data to set the prior later
    if i==1
        range_data = range(Y);
    else
        range_data = max(range_data,range(Y));
    end
    % get the sufficient stats
    Gamma = GammaInit((1:size(Y,1))+tsum,:); tsum = tsum + size(Y,1);
    for k=1:K
        XG = XX' .* repmat(Gamma(:,k)',size(XX,2),1);
        subj_m_init(:,:,k) = subj_m_init(:,:,k) + XG * Y;
        subj_gram_init(:,:,k) = subj_gram_init(:,:,k) + XG * XX;
    end
end

% init hmm and set priors
hmm = struct('train',struct());
hmm.K = K;
hmm.train = options;
hmm = hmmhsinit(hmm);
hmm = initspriors(hmm,range_data);
hmm.train.ndim = ndim;
hmm.train.active = ones(1,K);
hmm.train.Sind = Sind; 

% Compute W
for k = 1:K
    % this is regardless the choice of the cov.matrix - if full, we need to
    % recompute S_W/S_iW later
    iS_W = zeros(ndim,npred,npred);
    S_W = zeros(ndim,npred,npred);
    hmm.state(k).W.Mu_W = zeros(npred,ndim);
    for n = 1:ndim
        iS_W(n,:,:) = subj_gram_init(:,:,k) + eps*eye(npred);
        S_W(n,:,:) = inv(permute(iS_W(n,:,:),[2 3 1]));
        hmm.state(k).W.Mu_W(:,n) = permute(S_W(n,:,:),[2 3 1]) * subj_m_init(:,n,k);
    end
end

% Collect sufficient statistics for Omega
tsum = 0; 
for i = 1:N
    % read data
    [~,XX,Y] = loadfile(Xin{i},T{i},options);
    Gamma = GammaInit((1:size(Y,1))+tsum,:); 
    tsum = tsum + size(Y,1);
    for k = 1:K
        if strcmp(options.covtype,'diag')
            if ~isempty(hmm.state(k).W.Mu_W)
                e = (Y - XX * hmm.state(k).W.Mu_W).^2;
            else
                e = Y.^2;
            end
            subj_err_init(1,regressed,k) = subj_err_init(1,regressed,k) + sum( repmat(Gamma(:,k),1,ndim) .* e(:,regressed) ) / 2;
            subj_time_init(k) = subj_time_init(k) + sum(Gamma(:,k)) / 2;
        elseif strcmp(options.covtype,'full')
            if ~isempty(hmm.state(k).W.Mu_W)
                e = Y - XX * hmm.state(k).W.Mu_W;
            else
                e = Y;
            end
            subj_err_init(:,:,k) = subj_err_init(:,:,k) + (e' .* repmat(Gamma(:,k)',ndim,1)) * e;
            subj_time_init(k) = subj_time_init(k) + sum(Gamma(:,k));
        elseif strcmp(options.covtype,'uniquediag')
            if ~isempty(hmm.state(k).W.Mu_W)
                e = (Y - XX * hmm.state(k).W.Mu_W).^2;
            else
                e = Y.^2;
            end
            subj_err_init(regressed) = subj_err_init(regressed) + sum( repmat(Gamma(:,k),1,ndim) .* e(:,regressed) ) / 2;
            subj_time_init = subj_time_init + sum(Gamma(:,k)) / 2;
        else
            if ~isempty(hmm.state(k).W.Mu_W)
                e = Y - XX * hmm.state(k).W.Mu_W;
            else
                e = Y;
            end
            subj_err_init = subj_err_init + (e' .* repmat(Gamma(:,k)',ndim,1)) * e;
            subj_time_init = subj_time_init + sum(Gamma(:,k));
        end
    end
end
if (strcmp(options.covtype,'uniquefull') || strcmp(options.covtype,'uniquediag'))
    hmm.Omega.Gam_shape = subj_time_init + hmm.prior.Omega.Gam_shape;
    hmm.Omega.Gam_rate = subj_err_init + hmm.prior.Omega.Gam_rate;
end

% create the states
for k = 1:K
    if strcmp(options.covtype,'full')
        state = state_snew( ...
            sum(subj_err_init(:,:,k),3) + hmm.state(k).prior.Omega.Gam_rate, ...
            sum(subj_time_init(k)) + hmm.state(k).prior.Omega.Gam_shape, ...
            sum(subj_gram_init(:,:,k),3) + 0.01 * eye(npred), ...
            sum(subj_m_init(:,:,k),3),options.covtype,Sind);
    elseif strcmp(options.covtype,'diag')
        state = state_snew( ...
            sum(subj_err_init(:,k),2)' + hmm.state(k).prior.Omega.Gam_rate, ...
            sum(subj_time_init(k)) + hmm.state(k).prior.Omega.Gam_shape, ...
            sum(subj_gram_init(:,:,k),3) + 0.01 * eye(npred), ...
            sum(subj_m_init(:,:,k),3),options.covtype,Sind);
    else
        state = state_snew(hmm.Omega.Gam_rate,...
            hmm.Omega.Gam_shape,...
            sum(subj_gram_init(:,:,k),3) + 0.01 * eye(npred),...
            sum(subj_m_init(:,:,k),3),options.covtype,Sind);
    end
    hmm.state(k).W = state.W;
    hmm.state(k).Omega = state.Omega;
end

% distribution of sigma and alpha, variances of the MAR coeff distributions
if ~isempty(options.orders)
    if pcaprec
        hmm = updateBeta(hmm);
    else
        for k=1:K
            hmm.state(k).alpha.Gam_shape = hmm.state(k).prior.alpha.Gam_shape;
            hmm.state(k).alpha.Gam_rate = hmm.state(k).prior.alpha.Gam_rate;
        end
        hmm = updateSigma(hmm);
        hmm = updateAlpha(hmm);
    end
end

% compute Xi, P, Pi, ...
tsum = 0; 
for i = 1:N
    Ti = T{i} - hmm.train.order;
    if length(options.embeddedlags)>1
        Ti = Ti - (max(options.embeddedlags) + max(-options.embeddedlags));
    end
    Gamma = GammaInit((1:sum(Ti))+tsum,:); tsum = tsum + sum(Ti);
    Xi = zeros(size(Gamma,1)-length(Ti),K^2);
    for jj=1:length(Ti)
        sTi = sum(Ti(1:jj-1)); sTi2 = sTi - (jj-1); 
        for j=1:Ti(jj)-1
            t = Gamma(sTi+j,:)' * Gamma(sTi+j+1,:);
            Xi(sTi2+j,:)=t(:)'/sum(t(:));
        end
    end
    Xi = reshape(Xi,size(Xi,1),K,K);
    for jj=1:length(Ti)
        t = sum(Ti(1:jj-1)) + 1;
        Dir_alpha_init(:,i) = Dir_alpha_init(:,i) + Gamma(t,:)';
    end
    Dir2d_alpha_init(:,:,i) = squeeze(sum(Xi,1));
end
hmm.Dir_alpha = sum(Dir_alpha_init,2)' + hmm.prior.Dir_alpha;
hmm.Dir2d_alpha = sum(Dir2d_alpha_init,3) + hmm.prior.Dir2d_alpha;
[hmm.P,hmm.Pi] = computePandPi(hmm.Dir_alpha,hmm.Dir2d_alpha);
          
% compute free energy  
tsum = 0; 
for i = 1:N    
    % read data
    [~,XX,Y,Ti] = loadfile(Xin{i},T{i},options);
    Gamma = GammaInit((1:size(Y,1))+tsum,:); 
    tsum = tsum + size(Y,1);
    Xi = zeros(size(Gamma,1),K^2);
    for j=1:Ti-1-hmm.train.order
        t = Gamma(j,:)' * Gamma(j+1,:);
        Xi(j,:)=t(:)'/sum(t(:));
        Xi = reshape(Xi,size(Xi,1),K,K);
    end
    loglik_init(i) = -evalfreeenergy([],Ti,Gamma,[],hmm,Y,XX,[0 1 0 0 0]); % data LL
    subjfe_init(i,1:2) = evalfreeenergy([],Ti,Gamma,Xi,hmm,[],[],[1 0 1 0 0]); % Gamma entropy&LL
end
subjfe_init(:,3) = evalfreeenergy([],[],[],[],hmm,[],[],[0 0 0 1 0]) / N; % "share" P and Pi KL
statekl_init = sum(evalfreeenergy([],[],[],[],hmm,[],[],[0 0 0 0 1])); % state KL
fe = - sum(loglik_init) + sum(subjfe_init(:)) + statekl_init;

info.Dir2d_alpha = Dir2d_alpha_init; 
info.Dir_alpha = Dir_alpha_init;
info.subjfe = subjfe_init;
info.loglik = loglik_init;
info.statekl = statekl_init;
info.fehist = (-sum(info.loglik) + sum(info.statekl) + sum(sum(info.subjfe)));

if options.BIGverbose
    fprintf('Init, free energy = %g \n',fe);
end

end


% define priors
function hmm = initspriors(hmm,r)

ndim = length(r);
rangresiduals2 = (r/2).^2;
pcapred = hmm.train.pcapred>0;
if isfield(hmm.train,'B'), Q = size(hmm.train.B,2);
else Q = ndim; end
if pcapred, M = size(hmm.train.V,2); end

for k=1:hmm.K
    if isfield(hmm.train,'state') && isfield(hmm.train.state(k),'train') && ~isempty(hmm.train.state(k).train)
        train = hmm.state(k).train;
    else
        train = hmm.train;
    end
    orders = formorders(train.order,train.orderoffset,train.timelag,train.exptimelag);
    
    if (strcmp(train.covtype,'diag') || strcmp(train.covtype,'full')) && pcapred
        defstateprior(k)=struct('beta',[],'Omega',[],'Mean',[]);
    elseif (strcmp(train.covtype,'diag') || strcmp(train.covtype,'full')) && ~pcapred
        defstateprior(k)=struct('sigma',[],'alpha',[],'Omega',[],'Mean',[]);
    elseif (strcmp(train.covtype,'uniquediag') || strcmp(train.covtype,'uniquefull')) && pcapred
        defstateprior(k)=struct('beta',[],'Mean',[]);
    else
        defstateprior(k)=struct('sigma',[],'alpha',[],'Mean',[]);
    end
    if pcapred
        defstateprior(k).beta = struct('Gam_shape',[],'Gam_rate',[]);
        defstateprior(k).beta.Gam_shape = 0.1*ones(M,ndim); %+ 0.05*eye(ndim);
        defstateprior(k).beta.Gam_rate = 0.1*ones(M,ndim);%  + 0.05*eye(ndim);
    else
        if (~isfield(train,'uniqueAR') || ~train.uniqueAR)
            defstateprior(k).sigma = struct('Gam_shape',[],'Gam_rate',[]);
            defstateprior(k).sigma.Gam_shape = 0.1*ones(Q,ndim); %+ 0.05*eye(ndim);
            defstateprior(k).sigma.Gam_rate = 0.1*ones(Q,ndim);%  + 0.05*eye(ndim);
        end
        if ~isempty(orders)
            defstateprior(k).alpha = struct('Gam_shape',[],'Gam_rate',[]);
            defstateprior(k).alpha.Gam_shape = 0.1;
            defstateprior(k).alpha.Gam_rate = 0.1*ones(1,length(orders));
        end
    end
    if ~train.zeromean
        defstateprior(k).Mean = struct('Mu',[],'iS',[]);
        defstateprior(k).Mean.Mu = zeros(ndim,1);
        defstateprior(k).Mean.S = rangresiduals2';
        defstateprior(k).Mean.iS = 1./rangresiduals2';
    end
    if strcmp(train.covtype,'full')
        defstateprior(k).Omega.Gam_rate = diag(r);
        defstateprior(k).Omega.Gam_shape = ndim+0.1-1;
    elseif strcmp(train.covtype,'diag')
        defstateprior(k).Omega.Gam_rate = 0.5 * r;
        defstateprior(k).Omega.Gam_shape = 0.5 * (ndim+0.1-1);
    end
    
end

if strcmp(hmm.train.covtype,'uniquefull')
    hmm.prior.Omega.Gam_shape = ndim+0.1-1;
    hmm.prior.Omega.Gam_rate = diag(r);
elseif strcmp(hmm.train.covtype,'uniquediag')
    hmm.prior.Omega.Gam_shape = 0.5 * (ndim+0.1-1);
    hmm.prior.Omega.Gam_rate = 0.5 * r;
end

% assigning default priors for observation models
if ~isfield(hmm,'state') || ~isfield(hmm.state,'prior')
    for k=1:hmm.K
        hmm.state(k).prior=defstateprior(k);
    end
else
    for k=1:hmm.K
        % prior not specified are set to default
        statepriorlist=fieldnames(defstateprior(k));
        fldname=fieldnames(hmm.state(k).prior);
        misfldname=find(~ismember(statepriorlist,fldname));
        for i=1:length(misfldname)
            priorval=getfield(defstateprior(k),statepriorlist{i});
            hmm.state(k).prior=setfield(hmm.state,k,'prior',statepriorlist{i}, ...
                priorval);
        end
    end
end

% moving the state options for convenience
for k=1:hmm.K
   if isfield(hmm.train,'state') && isfield(hmm.train.state(k),'train') ...
           && ~isempty(hmm.train.state(k).train)
       hmm.state(k).train = hmm.train.state(k).train;
   end
end
if isfield(hmm.train,'state')
   hmm.train = rmfield(hmm.train,'state');
end

end



