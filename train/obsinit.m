function [hmm, residuals, W0] = obsinit (data,T,hmm,Gamma)
%
% Initialise observation model in HMM
%
% INPUT
% data          observations - a struct with X (time series) and C (classes)
% T             length of series
% Gamma         p(state given X)
% hmm           hmm data structure
%
% OUTPUT
% hmm           estimated HMMMAR model
% residuals     in case we train on residuals, the value of those
% W0            global MAR estimation
%
% Author: Diego Vidaurre, OHBA, University of Oxford

if nargin<4, Gamma = []; end
do_HMM_pca = (hmm.train.lowrank > 0);
if ~do_HMM_pca
    [residuals,W0] =  getresiduals(data.X,T,hmm.train.Sind,hmm.train.maxorder,hmm.train.order,...
        hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag,hmm.train.zeromean);
else 
    residuals = []; W0 = [];
end
hmm = initpriors(data.X,T,hmm,residuals);
hmm = initpost(data.X,T,hmm,residuals,Gamma);

end


function hmm = initpriors(X,T,hmm,residuals)
% define priors

ndim = size(X,2);
rangresiduals2 = (range(residuals)/2).^2;
if isfield(hmm.train,'B'), Q = size(hmm.train.B,2); 
else Q = ndim; 
end
pcapred = hmm.train.pcapred>0;
if pcapred, M = hmm.train.pcapred; end
p = hmm.train.lowrank; do_HMM_pca = (p > 0);

for k = 1:hmm.K

    train = hmm.train;
    orders = train.orders;
    %train.orders = formorders(train.order,train.orderoffset,train.timelag,train.exptimelag);

    if (strcmp(train.covtype,'diag') || strcmp(train.covtype,'full')) && pcapred
        defstateprior(k)=struct('beta',[],'Omega',[],'Mean',[]);
    elseif do_HMM_pca
        defstateprior(k)=struct('Omega',[]); %'beta',[]
    elseif (strcmp(train.covtype,'diag') || strcmp(train.covtype,'full')) 
        defstateprior(k)=struct('sigma',[],'alpha',[],'Omega',[],'Mean',[]);
    elseif (strcmp(train.covtype,'uniquediag') || strcmp(train.covtype,'uniquefull')) && pcapred
        defstateprior(k)=struct('beta',[],'Mean',[]);
    elseif isfield(train,'distribution') && strcmp(train.distribution,'logistic')
        defstateprior(k)=struct('sigma',[],'alpha',[]);
    else
        defstateprior(k)=struct('sigma',[],'alpha',[],'Mean',[]);
    end
    
    if do_HMM_pca
        %defstateprior(k).beta = struct('Gam_shape',[],'Gam_rate',[]);
        %defstateprior(k).beta.Gam_shape = 0.1; 
        %defstateprior(k).beta.Gam_rate = 0.1 * ones(1,p);
    elseif pcapred
        defstateprior(k).beta = struct('Gam_shape',[],'Gam_rate',[]);
        defstateprior(k).beta.Gam_shape = 0.1; %+ 0.05*eye(ndim);
        defstateprior(k).beta.Gam_rate = 0.1 * ones(M,ndim);%  + 0.05*eye(ndim);
    else
        if ~train.uniqueAR && isempty(train.prior)
            defstateprior(k).sigma = struct('Gam_shape',[],'Gam_rate',[]);
            defstateprior(k).sigma.Gam_shape = 0.1*ones(Q,ndim); %+ 0.05*eye(ndim);
            defstateprior(k).sigma.Gam_rate = 0.1*ones(Q,ndim);%  + 0.05*eye(ndim);
        end
        if ~isempty(orders) && isempty(train.prior)
            defstateprior(k).alpha = struct('Gam_shape',[],'Gam_rate',[]);
            defstateprior(k).alpha.Gam_shape = 0.1;
            defstateprior(k).alpha.Gam_rate = 0.1*ones(1,length(orders));
        end
        if  isfield(train,'distribution') && strcmp(train.distribution,'logistic') && isempty(train.prior)
            defstateprior(k).alpha = struct('Gam_shape',[],'Gam_rate',[]);
            defstateprior(k).alpha.Gam_shape = 0.1;
            defstateprior(k).alpha.Gam_rate = 0.1;
        end
    end
    if ~train.zeromean
        defstateprior(k).Mean = struct('Mu',[],'iS',[]);
        defstateprior(k).Mean.Mu = zeros(ndim,1);
        defstateprior(k).Mean.S = rangresiduals2';
        defstateprior(k).Mean.iS = 1./rangresiduals2';
    end
    if isempty(hmm.train.priorcov_rate) 
        priorcov_rate = rangeerror(X,T,residuals,orders,hmm.train);
    else
        priorcov_rate = hmm.train.priorcov_rate * ones(1,ndim);
    end
    if strcmp(train.covtype,'full')
        defstateprior(k).Omega.Gam_rate = diag(priorcov_rate);
        defstateprior(k).Omega.Gam_shape = ndim+0.1-1;
    elseif strcmp(train.covtype,'diag')
        defstateprior(k).Omega.Gam_rate = 0.5 * priorcov_rate;
        defstateprior(k).Omega.Gam_shape = 0.5 * (ndim+0.1-1);
    end
    
end

if strcmp(hmm.train.covtype,'uniquefull')
    hmm.prior.Omega.Gam_shape = ndim+0.1-1;
    hmm.prior.Omega.Gam_rate = diag(priorcov_rate);
elseif do_HMM_pca
    hmm.prior.Omega.Gam_shape = 0.5 * (ndim+0.1-1);
    hmm.prior.Omega.Gam_rate = 0.5 * median(priorcov_rate);
elseif strcmp(hmm.train.covtype,'uniquediag') 
    hmm.prior.Omega.Gam_shape = 0.5 * (ndim+0.1-1);
    hmm.prior.Omega.Gam_rate = 0.5 * priorcov_rate;     
end

% assigning default priors for observation models
if ~isfield(hmm,'state') || ~isfield(hmm.state,'prior')
    for k = 1:hmm.K
        hmm.state(k).prior = defstateprior(k);
    end
else
    for k = 1:hmm.K
        % prior not specified are set to default
        statepriorlist = fieldnames(defstateprior(k));
        fldname = fieldnames(hmm.state(k).prior);
        misfldname = find(~ismember(statepriorlist,fldname));
        for i = 1:length(misfldname)
            priorval = getfield(defstateprior(k),statepriorlist{i});
            hmm.state(k).prior = setfield(hmm.state,k,'prior',statepriorlist{i}, ...
                priorval);
        end
    end
end

end


function hmm = initpost(X,T,hmm,residuals,Gamma)
% Initialising the posteriors

Tres = sum(T) - length(T)*hmm.train.maxorder;
ndim = size(X,2);
K = hmm.K;
S = hmm.train.S==1; regressed = sum(S)>0;
hmm.train.active = ones(1,K);
if isfield(hmm.train,'B'), B = hmm.train.B; Q = size(B,2);
else Q = ndim; end
pcapred = hmm.train.pcapred>0;
p = hmm.train.lowrank; do_HMM_pca = (p > 0);

% initial random values for the states - multinomial
Gammasum = sum(Gamma);
setxx; % build XX and get orders

% W
for k = 1:K
    setstateoptions;
    if pcapred, npred = hmm.train.pcapred;
    else npred = Q*length(orders);
    end
    hmm.state(k).W = struct('Mu_W',[],'S_W',[]);
  
    if do_HMM_pca || order>0 || ~train.zeromean || ...
            (isfield(train,'distribution') && strcmp(train.distribution,'logistic'))
        if train.uniqueAR || ndim==1 % it is assumed that order>0 and cov matrix is diagonal
            XY = zeros(npred+(~train.zeromean),1);
            XGX = zeros(npred+(~train.zeromean));
            for n = 1:ndim
                ind = n:ndim:size(XX,2);
                XGX = XGX + XXGXX{k}(ind,ind);
                XY = XY + (XX(:,ind)' .* repmat(Gamma(:,k)',length(ind),1)) * residuals(:,n);
            end
            if ~isempty(train.prior)
                hmm.state(k).W.S_W = inv(train.prior.iS + XGX);
                hmm.state(k).W.Mu_W = hmm.state(k).W.S_W * (XY + train.prior.iSMu); % order by 1
            else
                %hmm.state(k).W.S_W = inv(0.1 * mean(trace(XGX)) * eye(length(orders)) + XGX);
                hmm.state(k).W.S_W = inv(0.01 * eye(npred+(~train.zeromean)) + XGX);
                hmm.state(k).W.Mu_W = hmm.state(k).W.S_W * XY; % order by 1
            end
            
        elseif do_HMM_pca
            weights = Gamma(:,k); weights(weights==0) = eps; 
            hmm.state(k).W.Mu_W = pca(XX,'NumComponents',p,'Weights',weights,'Centered',false);
            Xpca = XX * hmm.state(k).W.Mu_W;
            hmm.state(k).W.iS_W = zeros(ndim,p,p);
            hmm.state(k).W.S_W = zeros(ndim,p,p);      
            for n = 1:ndim
                hmm.state(k).W.iS_W(n,:,:) = (Xpca' .* repmat(Gamma(:,k)',p,1)) * Xpca + 0.01*eye(p);
                hmm.state(k).W.S_W(n,:,:) =  diag(1 ./ diag(permute(hmm.state(k).W.iS_W(n,:,:),[2 3 1])));
            end
            
        elseif strcmp(train.covtype,'uniquediag') || strcmp(train.covtype,'diag') || ...
                (isfield(train,'distribution') && strcmp(train.distribution,'logistic'))
            hmm.state(k).W.Mu_W = zeros((~train.zeromean)+npred,ndim);
            hmm.state(k).W.iS_W = zeros(ndim,(~train.zeromean)+npred,(~train.zeromean)+npred);
            hmm.state(k).W.S_W = zeros(ndim,(~train.zeromean)+npred,(~train.zeromean)+npred);
            for n = 1:ndim
                ndim_n = sum(S(:,n)>0);
                if ndim_n==0, continue; end
                hmm.state(k).W.iS_W(n,Sind(:,n),Sind(:,n)) = ...
                    XXGXX{k}(Sind(:,n),Sind(:,n)) + 0.01*eye(sum(Sind(:,n))) ;
                hmm.state(k).W.S_W(n,Sind(:,n),Sind(:,n)) = ...
                    inv(permute(hmm.state(k).W.iS_W(n,Sind(:,n),Sind(:,n)),[2 3 1]));
                hmm.state(k).W.Mu_W(Sind(:,n),n) = ...
                    (( permute(hmm.state(k).W.S_W(n,Sind(:,n),Sind(:,n)),[2 3 1])...
                    * XX(:,Sind(:,n))') .* repmat(Gamma(:,k)',sum(Sind(:,n)),1)) * residuals(:,n);
            end
        else
            if all(S(:))==1
                gram = kron(XXGXX{k},eye(ndim));
                hmm.state(k).W.iS_W = gram + 0.01*eye(size(gram,1));
                hmm.state(k).W.S_W = inv( hmm.state(k).W.iS_W );
                hmm.state(k).W.Mu_W = (( XXGXX{k} \ XX' ) .* ...
                    repmat(Gamma(:,k)',(~train.zeromean)+npred,1)) * residuals;
            else
                regressed = sum(S,1)>0; % dependent variables, Y
                index_iv = sum(S,2)>0; % independent variables, X
                % note that XXGXX is invalid if any S==0:
                hmm.state(k).W.iS_W = zeros(length(S(:)));
                hmm.state(k).W.S_W = zeros(length(S(:)));
                temp1 = bsxfun(@times,XX(:,index_iv),Gamma(:,k));
                gram = kron(eye(sum(regressed)),temp1'*XX(:,index_iv));
                hmm.state(k).W.iS_W(S(:),S(:)) = gram + 0.01*eye(size(gram,1));
                hmm.state(k).W.S_W(S(:),S(:)) = inv( hmm.state(k).W.iS_W(S(:),S(:)) );
                hmm.state(k).W.Mu_W = zeros(size(S));
                % intialise to OLS estimate:
                hmm.state(k).W.Mu_W(S) = ...
                    pinv(residuals(Gamma(:,k)>0.5,index_iv))*residuals(Gamma(:,k)>0.5,regressed);
            end
        end
    end
end

% Omega
if strcmp(hmm.train.covtype,'uniquediag') && hmm.train.uniqueAR
    hmm.Omega.Gam_rate = hmm.prior.Omega.Gam_rate;
    for k = 1:K
        XW = zeros(size(XX,1),ndim);
        for n = 1:ndim
            ind = n:ndim:size(XX,2);
            XW(:,n) = XX(:,ind) * hmm.state(k).W.Mu_W;
        end
        e = (residuals - XW).^2;
        hmm.Omega.Gam_rate = hmm.Omega.Gam_rate + ...
            0.5 * sum( repmat(Gamma(:,k),1,ndim) .* e );
    end
    hmm.Omega.Gam_shape = hmm.prior.Omega.Gam_shape + Tres / 2;
    
elseif do_HMM_pca
    hmm.Omega.Gam_rate = hmm.prior.Omega.Gam_rate;
    hmm.Omega.Gam_shape = hmm.prior.Omega.Gam_shape;
    v = hmm.Omega.Gam_rate / hmm.Omega.Gam_shape;
    for k = 1:K
        W = hmm.state(k).W.Mu_W;
        M = W' * W + v * eye(p); % posterior dist of the precision matrix 
        omega_i = mean(diag(XXGXX{k} - XXGXX{k} * W * (M \ W')));
        %e = sum(repmat(Gamma(:,k),1,ndim) .* (XX - XX * W * W').^2,2);
        hmm.Omega.Gam_rate_state(k) = 0.5 * omega_i; %sum(e);
        hmm.Omega.Gam_rate = hmm.Omega.Gam_rate + hmm.Omega.Gam_rate_state(k);
        hmm.Omega.Gam_shape_state(k) = 0.5 * Gammasum(k);% * ndim;
        hmm.Omega.Gam_shape = hmm.Omega.Gam_shape + hmm.Omega.Gam_shape_state(k);
    end
    
elseif strcmp(hmm.train.covtype,'uniquediag')
    hmm.Omega.Gam_shape = hmm.prior.Omega.Gam_shape + Tres / 2;
    hmm.Omega.Gam_rate = zeros(1,ndim);
    hmm.Omega.Gam_rate(regressed) = hmm.prior.Omega.Gam_rate(regressed);
    for k = 1:hmm.K
        if ~isempty(hmm.state(k).W.Mu_W(:,regressed))
            e = residuals(:,regressed) - XX * hmm.state(k).W.Mu_W(:,regressed);
        else
            e = residuals(:,regressed);
        end
        hmm.Omega.Gam_rate(regressed) = hmm.Omega.Gam_rate(regressed) +  ...
            0.5 * sum( repmat(Gamma(:,k),1,sum(regressed)) .* e.^2 );
    end
    
elseif strcmp(hmm.train.covtype,'uniquefull')
    hmm.Omega.Gam_shape = hmm.prior.Omega.Gam_shape + Tres;
    hmm.Omega.Gam_rate = zeros(ndim,ndim); hmm.Omega.Gam_irate = zeros(ndim,ndim);
    hmm.Omega.Gam_rate(regressed,regressed) = hmm.prior.Omega.Gam_rate(regressed,regressed);
    for k = 1:hmm.K
        if ~isempty(hmm.state(k).W.Mu_W(:,regressed))
            e = residuals(:,regressed) - XX * hmm.state(k).W.Mu_W(:,regressed);
        else
            e = residuals(:,regressed);
        end
        hmm.Omega.Gam_rate(regressed,regressed) = hmm.Omega.Gam_rate(regressed,regressed) +  ...
            (e' .* repmat(Gamma(:,k)',sum(regressed),1)) * e;
    end
    hmm.Omega.Gam_irate(regressed,regressed) = inv(hmm.Omega.Gam_rate(regressed,regressed));   
    
elseif ~isfield(hmm.train,'distribution') || ~strcmp(hmm.train.distribution,'logistic') % state dependent
    for k = 1:K
        setstateoptions;
        if train.uniqueAR
            XW = zeros(size(XX,1),ndim);
            for n=1:ndim
                ind = n:ndim:size(XX,2);
                XW(:,n) = XX(:,ind) * hmm.state(k).W.Mu_W;
            end
            e = (residuals - XW).^2;
            hmm.state(k).Omega.Gam_rate = hmm.state(k).prior.Omega.Gam_rate + ...
                0.5* sum( repmat(Gamma(:,k),1,ndim) .* e );
            hmm.state(k).Omega.Gam_shape = hmm.state(k).prior.Omega.Gam_shape + Gammasum(k) / 2;

        elseif strcmp(train.covtype,'diag')
            if ~isempty(hmm.state(k).W.Mu_W)
                e = (residuals(:,regressed) - XX * hmm.state(k).W.Mu_W(:,regressed)).^2;
            else
                e = residuals(:,regressed).^2;
            end
            hmm.state(k).Omega.Gam_rate = zeros(1,ndim);
            hmm.state(k).Omega.Gam_rate(regressed) = hmm.state(k).prior.Omega.Gam_rate(regressed) + ...
                sum( repmat(Gamma(:,k),1,sum(regressed)) .* e ) / 2;
            hmm.state(k).Omega.Gam_shape = hmm.state(k).prior.Omega.Gam_shape + Gammasum(k) / 2;
            
        else % full
            if ~isempty(hmm.state(k).W.Mu_W)
                e = residuals(:,regressed) - XX * hmm.state(k).W.Mu_W(:,regressed);
            else
                e = residuals(:,regressed);
            end
            hmm.state(k).Omega.Gam_shape = hmm.state(k).prior.Omega.Gam_shape + Gammasum(k);
            hmm.state(k).Omega.Gam_rate = zeros(ndim,ndim); hmm.state(k).Omega.Gam_irate = zeros(ndim,ndim);
            hmm.state(k).Omega.Gam_rate(regressed,regressed) =  ...
                hmm.state(k).prior.Omega.Gam_rate(regressed,regressed) +  ...
                (e' .* repmat(Gamma(:,k)',sum(regressed),1)) * e;
            hmm.state(k).Omega.Gam_irate(regressed,regressed) = ...
                inv(hmm.state(k).Omega.Gam_rate(regressed,regressed));
        end
    end
    
end

%%% Priors
if ~pcapred && ~do_HMM_pca
    for k = 1:K
        if hmm.train.order>0 && isempty(hmm.train.prior)
            hmm.state(k).alpha.Gam_shape = hmm.state(k).prior.alpha.Gam_shape;
            hmm.state(k).alpha.Gam_rate = hmm.state(k).prior.alpha.Gam_rate;
        end
    end
    %%% sigma - channel x channel coefficients
    hmm = updateSigma(hmm);
    %%% alpha - one per order
    hmm = updateAlpha(hmm);
    if isfield(train,'distribution') && strcmp(train.distribution,'logistic')
        if train.logisticYdim>1
            for k = 1:train.K
                hmm.state(k).alpha.Gam_rate = ...
                    repmat(hmm.state(k).alpha.Gam_rate(1:ndim_n,end),1,train.logisticYdim);
            end
        end
    end
elseif pcapred
    for k = 1:K
        hmm.state(k).beta.Gam_shape = hmm.state(k).prior.beta.Gam_shape + 0.5 * ndim;
        hmm.state(k).beta.Gam_rate = hmm.state(k).prior.beta.Gam_rate + ...
            sum(hmm.state(k).W.Mu_W.^2);
    end
end

end



