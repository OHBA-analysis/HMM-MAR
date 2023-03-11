function [FrEn,avLL] = evalfreeenergy(X,T,Gamma,Xi,hmm,residuals,XX,todo,scale)
% Computes the Free Energy of an HMM depending on observation model
%
% INPUT
% X            observations
% T            length of series
% Gamma        probability of states conditioned on data
% Xi           joint probability of past and future states conditioned on data
% hmm          hmm structure
% residuals    in case we train on residuals, the value of those.
%
% OUTPUT
% FrEn         the variational free energy, separated in different terms:
%                   element 1: Gamma Entropy
%                   element 2: data negative loglikelihood
%                   element 3: Gamma negative loglikelihood
%                   element 4: KL for initial and transition probabilities
%                   elements 5: KL for the state parameters
% avLL         log likelihood of the observed data, per trial
%
% Author: Diego Vidaurre, OHBA, University of Oxford

if nargin<8 || isempty(todo), todo = ones(1,5); end
if nargin<9, scale = []; end

if isfield(hmm.train,'distribution')
    distname = hmm.train.distribution;
else
    distname = 'Gaussian';
end

p = hmm.train.lowrank; do_HMM_pca = (p > 0);
K = hmm.K;

if (nargin<7 || isempty(XX)) && todo(2)==1
    setxx; % build XX and get orders
end
if strcmp(distname,'Gaussian')
    if isfield(hmm.state(1),'W') && ~isempty(hmm.state(1).W.Mu_W)
        if do_HMM_pca
            ndim = size(hmm.state(1).W.Mu_W,1);
        else
            ndim = size(hmm.state(1).W.Mu_W,2);
        end
    else
        ndim = size(hmm.state(1).Omega.Gam_rate,2);
    end
elseif strcmp(distname,'bernoulli')
    ndim = size(hmm.state(1).W.a,2); %bernoulli
elseif strcmp(distname,'poisson')
    ndim = size(hmm.state(1).W.W_shape,2); %poisson
elseif strcmp(distname,'logistic')
    if isfield(hmm.state(1),'W')
        ndim = size(hmm.state(1).W.Mu_W,2);
    else
        ndim = size(hmm.state(1).Omega.Gam_rate,2);
    end
    regressed = sum(hmm.train.S,1)>0;
    % Set Y (unidimensional for now) and X: 
    Xdim = size(XX,2) - hmm.train.logisticYdim;
    X = XX(:,1:Xdim);
    Y = residuals;
end

if isfield(hmm.train,'B'), Q = size(hmm.train.B,2);
else, Q = ndim; end
pcapred = hmm.train.pcapred>0;
if pcapred, M = hmm.train.pcapred; end

N = length(T); 
Tres = sum(T) - N*hmm.train.maxorder;
setstateoptions; 
ltpi = sum(regressed)/2 * log(2*pi);

if sum(~regressed)>1 && ~isempty(X)
    %catch special case where regressors do not vary within trials:
    for t = 1:length(T)
        varcheck(t,:) = var(X(sum(T(1:t-1))+[1:T(t)-1],~regressed));
    end
    trialwiseregression = all(varcheck(:)==0);
else
    trialwiseregression = false;
end

if ~do_HMM_pca && (nargin<6 || isempty(residuals)) && todo(2)==1
    if ~isfield(hmm.train,'Sind')
        hmm.train.Sind = formindexes(orders,hmm.train.S) == 1;
    end
    if ~hmm.train.zeromean, hmm.train.Sind = [true(1,ndim); hmm.train.Sind]; end
    residuals =  getresiduals(X,T,hmm.train.S,hmm.train.maxorder,hmm.train.order,...
        hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag,hmm.train.zeromean);
end

if isempty(scale)
    
    % Entropy of Gamma
    Entr = [];
    if todo(1)==1
        if hmm.train.acrosstrial_constrained
            Entr = GammaEntropy(Gamma(1:T(1)-order,:),Xi(1:T(1)-order-1,:,:),...
                T(1),hmm.train.maxorder);
        else
            Entr = GammaEntropy(Gamma,Xi,T,hmm.train.maxorder);
        end
    end
    
    % loglikelihood of Gamma
    avLLGamma = [];
    if todo(3)==1
        if hmm.train.acrosstrial_constrained
            avLLGamma = GammaavLL(hmm,Gamma(1:T(1)-order,:),...
                Xi(1:T(1)-order-1,:,:),T(1));
        else
            avLLGamma = GammaavLL(hmm,Gamma,Xi,T);
        end
        
    end
    
end

% P and Pi KL
KLdivTran = [];
if todo(4)==1
    KLdivTran = KLtransition(hmm);
end


% state KL
if strcmp(distname,'Gaussian')
    KLdiv = [];
    if todo(5)==1
        
        if do_HMM_pca
            OmegaKL = gamma_kl(hmm.Omega.Gam_shape,hmm.prior.Omega.Gam_shape, ...
                hmm.Omega.Gam_rate,hmm.prior.Omega.Gam_rate);
            KLdiv = [KLdiv OmegaKL];
        elseif strcmp(hmm.train.covtype,'uniquediag') || strcmp(hmm.train.covtype,'shareddiag')
            OmegaKL = 0;
            for n = 1:ndim
                if ~regressed(n), continue; end
                OmegaKL = OmegaKL + gamma_kl(hmm.Omega.Gam_shape,hmm.prior.Omega.Gam_shape, ...
                    hmm.Omega.Gam_rate(n),hmm.prior.Omega.Gam_rate(n));
            end
            KLdiv = [KLdiv OmegaKL];
        elseif strcmp(hmm.train.covtype,'uniquefull') || strcmp(hmm.train.covtype,'sharedfull')
            OmegaKL = wishart_kl(hmm.Omega.Gam_rate(regressed,regressed),...
                hmm.prior.Omega.Gam_rate(regressed,regressed), ...
                hmm.Omega.Gam_shape,hmm.prior.Omega.Gam_shape);
            KLdiv = [KLdiv OmegaKL];
        end
        
        for k = 1:K
            hs = hmm.state(k);
            pr = hmm.state(k).prior;
            WKL = 0;
            
            if ~isempty(hs.W.Mu_W)
                if train.uniqueAR || ndim==1
                    if ~isempty(train.prior)
                        prior_var = train.prior.S;
                        prior_mu = train.prior.Mu;
                    else
                        prior_prec = [];
                        if train.zeromean==0
                            prior_prec = hs.prior.Mean.iS;
                        end
                        if ~isempty(orders)
                            if pcapred
                                prior_prec = [prior_prec; (hs.beta.Gam_shape ./ hs.beta.Gam_rate)];
                                prior_mu = zeros(M + ~train.zeromean,1);
                            else
                                prior_prec = [prior_prec (hs.alpha.Gam_shape ./  hs.alpha.Gam_rate)];
                                prior_mu = zeros(length(orders) + ~train.zeromean,1);
                            end
                        else
                            prior_mu = 0;
                        end
                        prior_var = diag(1 ./ prior_prec);
                    end
                    if train.uniqueAR || ndim==1
                        WKL = gauss_kl(hs.W.Mu_W, prior_mu, hs.W.S_W, prior_var);
                    else
                        WKL = gauss_kl(hs.W.Mu_W, prior_mu, permute(hs.W.S_W,[2 3 1]), prior_var);
                    end
                    
                elseif do_HMM_pca
                    %for n = 1:p
                    %   prior_prec = hs.beta.Gam_shape ./ hs.beta.Gam_rate(n);
                    %   prior_var = diag(1 ./ prior_prec);
                    %   WKL = WKL + gauss_kl(hs.W.Mu_W(:,n),zeros(ndim,1), ...
                    %       diag(hs.W.S_W(:,n,n)), eye(ndim)*prior_var);
                    %end
                    
                elseif strcmp(train.covtype,'diag') || ...
                        strcmp(train.covtype,'uniquediag') || strcmp(train.covtype,'shareddiag')
                    for n = 1:ndim
                        if ~regressed(n), continue; end
                        prior_prec = [];
                        if train.zeromean==0
                            prior_prec = hs.prior.Mean.iS(n);
                        end
                        if ~isempty(orders)
                            if pcapred
                                prior_prec = [prior_prec; (hs.beta.Gam_shape ./ hs.beta.Gam_rate(:,n))];
                            else
                                ndim_n = sum(S(:,n));
                                alphamat = repmat( (hs.alpha.Gam_shape ./  hs.alpha.Gam_rate), ndim_n, 1);
                                prior_prec = [prior_prec; repmat(hs.sigma.Gam_shape(S(:,n)==1,n) ./ ...
                                    hs.sigma.Gam_rate(S(:,n)==1,n), length(orders), 1) .* alphamat(:)] ;
                            end
                        end
                        prior_var = diag(1 ./ prior_prec);
                        WKL = WKL + gauss_kl(hs.W.Mu_W(Sind(:,n),n),zeros(sum(Sind(:,n)),1), ...
                            permute(hs.W.S_W(n,Sind(:,n),Sind(:,n)),[2 3 1]), prior_var);
                    end
                    
                else % full or sharedfull
                    if all(S(:)==1)
                        prior_prec = [];
                        if train.zeromean==0
                            prior_prec = hs.prior.Mean.iS;
                        end
                        if ~isempty(orders)
                            if pcapred
                                betaterm = (hs.beta.Gam_shape ./ hs.beta.Gam_rate)';
                                prior_prec = [prior_prec; betaterm(:)];
                            else
                                sigmaterm = (hs.sigma.Gam_shape ./ hs.sigma.Gam_rate)';
                                sigmaterm = repmat(sigmaterm(:), length(orders), 1);
                                alphaterm = repmat( (hs.alpha.Gam_shape ./  hs.alpha.Gam_rate), ndim*Q, 1);
                                alphaterm = alphaterm(:);
                                prior_prec = [prior_prec; alphaterm .* sigmaterm];
                            end
                        end
                        prior_var = diag(1 ./ prior_prec);
                        mu_w = hs.W.Mu_W';
                        mu_w = mu_w(:);
                        WKL = gauss_kl(mu_w,zeros(length(mu_w),1), hs.W.S_W, prior_var);
                    else % only computed over selected terms
                        regressed = sum(S,1)>0; % dependent variables, Y
                        index_iv = sum(S,2)>0; % independent variables, X
                        prior_prec = [];
                        if train.zeromean==0
                            prior_prec = hs.prior.Mean.iS;
                        end
                        sigmaterm = (hs.sigma.Gam_shape(regressed,regressed) ./ ...
                            hs.sigma.Gam_rate(regressed,regressed))';
                        sigmaterm = repmat(sigmaterm(:), length(orders), 1);
                        alphaterm = repmat( (hs.alpha.Gam_shape ./  hs.alpha.Gam_rate), length(sigmaterm), 1);
                        alphaterm = alphaterm(:);
                        prior_prec = [prior_prec; alphaterm .* sigmaterm];
                        prior_var = diag(1 ./ prior_prec);
                        % instead, using ridge regression:
                        prior_var = 0.1 * eye(sum(regressed)*sum(index_iv));
                        mu_w = hs.W.Mu_W(index_iv,regressed);
                        mu_w = mu_w(:);
                        valid_dims = S(:)==1;
                        WKL = gauss_kl(mu_w,zeros(length(mu_w),1), hs.W.S_W(valid_dims,valid_dims), prior_var);
                    end
                end
            end
            
            OmegaKL = [];
            switch train.covtype
                case 'diag'
                    OmegaKL = 0;
                    for n = 1:ndim
                        if ~regressed(n), continue; end
                        OmegaKL = OmegaKL + gamma_kl(hs.Omega.Gam_shape,hs.prior.Omega.Gam_shape, ...
                            hs.Omega.Gam_rate(n),hs.prior.Omega.Gam_rate(n));
                    end
                case 'full'
                    try
                        OmegaKL = wishart_kl(hs.Omega.Gam_rate(regressed,regressed),...
                            hs.prior.Omega.Gam_rate(regressed,regressed), ...
                            hs.Omega.Gam_shape,hs.prior.Omega.Gam_shape);
                    catch
                        error(['Error computing kullback-leibler divergence of the cov matrix - ' ...
                            'Something strange with the data?'])
                    end
            end
            
            KLdiv = [KLdiv OmegaKL WKL];
            
            if do_HMM_pca
                %betaKL = 0;
                %for n = 1:p
                %    betaKL = betaKL + gamma_kl(hs.beta.Gam_shape,pr.beta.Gam_shape, ...
                %        hs.beta.Gam_rate(n),pr.beta.Gam_rate(n));
                %end
                %KLdiv = [KLdiv betaKL];
            elseif pcapred
                betaKL = 0;
                for n1 = 1:M
                    for n2 = 1:ndim
                        betaKL = betaKL + gamma_kl(hs.beta.Gam_shape(n1,n2),pr.beta.Gam_shape(n1,n2), ...
                            hs.beta.Gam_rate(n1,n2),pr.beta.Gam_rate(n1,n2));
                    end
                end
                KLdiv = [KLdiv betaKL];
            elseif ~do_HMM_pca
                sigmaKL = 0;
                if isempty(train.prior) && ~isempty(orders) && ~train.uniqueAR && ndim>1
                    for n1=1:Q
                        for n2=1:ndim
                            if (train.symmetricprior && n2<n1) || S(n1,n2)==0, continue; end
                            sigmaKL = sigmaKL + gamma_kl(hs.sigma.Gam_shape(n1,n2),pr.sigma.Gam_shape(n1,n2), ...
                                hs.sigma.Gam_rate(n1,n2),pr.sigma.Gam_rate(n1,n2));
                        end
                    end
                end
                alphaKL = 0;
                if isempty(train.prior) &&  ~isempty(orders)
                    for i=1:length(orders)
                        alphaKL = alphaKL + gamma_kl(hs.alpha.Gam_shape,pr.alpha.Gam_shape, ...
                            hs.alpha.Gam_rate(i),pr.alpha.Gam_rate(i));
                    end
                end
                KLdiv = [KLdiv sigmaKL alphaKL];
                
            end
        end
    end
elseif strcmp(distname,'bernoulli')
    KLdiv = 0;
    if todo(5)==1
        W_KL = 0;
        for k=1:K
            hs = hmm.state(k);
            pr = hmm.state(k).prior;
            for xd=1:ndim
                W_KL = W_KL + sum(beta_kl(hs.W.a(xd),pr.alpha.a, ...
                    hs.W.b(xd),pr.alpha.b));
            end
        end
        KLdiv = KLdiv + W_KL;      
    end
elseif strcmp(distname,'poisson')
    KLdiv = 0;
    if todo(5)==1
        W_KL=0;
        for k=1:K
            hs = hmm.state(k);
            pr = hmm.state(k).prior;
            for xd=1:ndim
                W_KL = W_KL + sum(gamma_kl(hs.W.W_shape(xd),pr.alpha.Gam_shape, ...
                    hs.W.W_rate(1),pr.alpha.Gam_rate));
            end
        end
        KLdiv = KLdiv + W_KL;      
    end
elseif strcmp(distname,'logistic')
    n = Xdim+1;
    if todo(5)==1
        W_mu0=zeros(Xdim,1);
        %W_sig0=eye(Xdim);
        alphaKL=[];
        beta_KL = [];
        for k=1:K
            hs=hmm.state(k);
            pr=hmm.state(k).prior;
            for ly = n:n+hmm.train.logisticYdim-1
                W_sig0 = diag(hs.alpha.Gam_shape ./ (hs.alpha.Gam_rate(1:Xdim,ly-Xdim)));
                beta_KL = [ beta_KL, gauss_kl(hs.W.Mu_W(1:Xdim,ly),W_mu0, ...
                    squeeze(hs.W.S_W(ly,1:Xdim,1:Xdim)),W_sig0)];
                alphaKL_st=zeros(Xdim,1);
                for xd=1:Xdim
                    alphaKL_st(xd) = sum(gamma_kl(hs.alpha.Gam_shape,pr.alpha.Gam_shape, ...
                        hs.alpha.Gam_rate(xd,ly-Xdim),pr.alpha.Gam_rate));
                end
                alphaKL = [alphaKL,sum(alphaKL_st)];
            end
        end
        KLdiv = sum(beta_KL) + sum(alphaKL);      
    end
end

% data log likelihood
if isempty(scale)
    savLL = [];
    if todo(2)==1
        if strcmp(distname,'Gaussian')
            avLL = zeros(Tres,1);
            if strcmp(hmm.train.covtype,'uniquediag') || strcmp(hmm.train.covtype,'shareddiag')
                ldetWishB = 0;
                PsiWish_alphasum = 0;
                for n = 1:ndim
                    if ~regressed(n), continue; end
                    ldetWishB = ldetWishB+0.5*log(hmm.Omega.Gam_rate(n));
                    PsiWish_alphasum = PsiWish_alphasum+0.5*psi(hmm.Omega.Gam_shape);
                end
                C = hmm.Omega.Gam_shape ./ hmm.Omega.Gam_rate;
                avLL = avLL + (-ltpi-ldetWishB+PsiWish_alphasum);
            elseif strcmp(hmm.train.covtype,'uniquefull') || strcmp(hmm.train.covtype,'sharedfull')
                ldetWishB = 0.5*logdet(hmm.Omega.Gam_rate(regressed,regressed));
                PsiWish_alphasum = 0;
                for n = 1:sum(regressed)
                    PsiWish_alphasum=PsiWish_alphasum+0.5*psi(hmm.Omega.Gam_shape/2+0.5-n/2);
                end
                C = hmm.Omega.Gam_shape * hmm.Omega.Gam_irate;
                avLL = avLL + (-ltpi-ldetWishB+PsiWish_alphasum);
            end
            
            for k = 1:K
                hs = hmm.state(k);
                
                if strcmp(train.covtype,'diag')
                    ldetWishB = 0;
                    PsiWish_alphasum = 0;
                    for n = 1:ndim
                        if ~regressed(n), continue; end
                        ldetWishB = ldetWishB+0.5*log(hs.Omega.Gam_rate(n));
                        PsiWish_alphasum = PsiWish_alphasum+0.5*psi(hs.Omega.Gam_shape);
                    end
                    C = hs.Omega.Gam_shape ./ hs.Omega.Gam_rate;
                    avLL = avLL + Gamma(:,k) * (-ltpi-ldetWishB+PsiWish_alphasum);
                    
                elseif strcmp(train.covtype,'full')
                    ldetWishB = 0.5*logdet(hs.Omega.Gam_rate(regressed,regressed));
                    PsiWish_alphasum=0;
                    for n = 1:sum(regressed)
                        PsiWish_alphasum = PsiWish_alphasum+0.5*psi(hs.Omega.Gam_shape/2+0.5-n/2);
                    end
                    C = hs.Omega.Gam_shape * hs.Omega.Gam_irate;
                    avLL = avLL + Gamma(:,k) * (-ltpi-ldetWishB+PsiWish_alphasum);
                end
                
                if do_HMM_pca
                    W = hmm.state(k).W.Mu_W;
                    v = hmm.Omega.Gam_rate / hmm.Omega.Gam_shape;
                    C = W * W' + v * eye(ndim);
                    ldetWishB = 0.5*logdet(C); PsiWish_alphasum = 0;
                    avLL = avLL + Gamma(:,k) * (-ltpi-ldetWishB+PsiWish_alphasum);
                    dist = - 0.5 * sum((XX / C) .* XX,2);
                else
                    meand = zeros(size(XX,1),sum(regressed)); % mean distance
                    if train.uniqueAR
                        for n = 1:ndim
                            ind = n:ndim:size(XX,2);
                            meand(:,n) = XX(:,ind) * hs.W.Mu_W;
                        end
                    elseif ~isempty(hs.W.Mu_W(:))
                        meand = XX * hs.W.Mu_W(:,regressed);
                    end
                    d = residuals(:,regressed) - meand;
                    if strcmp(train.covtype,'diag') || ...
                            strcmp(train.covtype,'uniquediag') || strcmp(hmm.train.covtype,'shareddiag')
                        Cd = bsxfun(@times, C(regressed), d)';
                    else
                        Cd = C(regressed,regressed) * d';
                    end
                    dist = zeros(Tres,1);
                    for n = 1:sum(regressed)
                        dist = dist - 0.5 * (d(:,n).*Cd(n,:)');
                    end
                end
                
                NormWishtrace = zeros(Tres,1); % covariance of the distance
                if ~isempty(hs.W.Mu_W)
                    switch train.covtype
                        case {'diag','uniquediag','shareddiag'}
                            if do_HMM_pca
                                %SW = eye(ndim) * trace(permute(hs.W.S_W(1,:,:),[2 3 1]));
                                %C = (hs.Omega.Gam_rate ./ hs.Omega.Gam_shape) * eye(ndim) + SW;
                                %iC = inv(C);
                                %for t = 1:T
                                %    NormWishtrace(t) = 0.5 * trace(iC * (XX(t,:)' * XX(t,:)) );
                                %end
                            else
                                for n = 1:ndim
                                    if ~regressed(n), continue; end
                                    if train.uniqueAR
                                        ind = n:ndim:size(XX,2);
                                        NormWishtrace = NormWishtrace + 0.5 * C(n) * ...
                                            sum( (XX(:,ind) * hs.W.S_W) .* XX(:,ind), 2);
                                    elseif ndim==1
                                        NormWishtrace = NormWishtrace + 0.5 * C(n) * ...
                                            sum( (XX(:,Sind(:,n)) * hs.W.S_W) ...
                                            .* XX(:,Sind(:,n)), 2);
                                    else
                                        NormWishtrace = NormWishtrace + 0.5 * C(n) * ...
                                            sum( (XX(:,Sind(:,n)) * ...
                                            permute(hs.W.S_W(n,Sind(:,n),Sind(:,n)),[2 3 1])) ...
                                            .* XX(:,Sind(:,n)), 2);
                                    end
                                end
                            end
                            
                        case {'full','uniquefull','sharedfull'}
                            if isempty(orders)
                                NormWishtrace = 0.5 * sum(sum(C .* hs.W.S_W));
                            else
                                if hmm.train.pcapred>0
                                    I = (0:hmm.train.pcapred+(~train.zeromean)-1) * ndim;
                                else
                                    I = (0:length(orders)*Q+(~train.zeromean)-1) * ndim;
                                end
                                if ~(all(S(:)==1) || ~trialwiseregression)
                                    [WishTrace,X_coded_vals] = computeWishTrace(hmm,regressed,XX,C,k);
                                end
                                
                                if all(S(:)==1) || ~trialwiseregression
                                    for n1 = 1:ndim
                                        if ~regressed(n1), continue; end
                                        index1 = I + n1; index1 = index1(Sind(:,n1));
                                        tmp = (XX(:,Sind(:,n1)) * hs.W.S_W(index1,:));
                                        for n2 = 1:ndim
                                            if ~regressed(n2), continue; end
                                            index2 = I + n2; index2 = index2(Sind(:,n2));
                                            NormWishtrace = NormWishtrace + 0.5 * C(n1,n2) * ...
                                                sum( tmp(:,index2) .* XX(:,Sind(:,n2)),2);
                                        end
                                    end
                                elseif ~isempty(WishTrace)
                                    Xval = XX(:,~regressed)*[2.^(1:sum(~regressed))]';
                                    for iR = 1:length(X_coded_vals)
                                        NormWishtrace(Xval==X_coded_vals(iR)) = WishTrace(iR);
                                    end
                                else
                                    % time varying regressors - this inference will be
                                    % exceedingly slow, is approximated here by
                                    % using first value at each trial
                                    validentries = logical(S(:));
                                    B_S = hmm.state(k).W.S_W(validentries,validentries);
                                    for iT=1:length(T)
                                        T_temp = T - order;
                                        t = sum(T_temp(1:iT-1))+1;
                                        NormWishtrace(t:t+T_temp(iT)-1) = trace(kron(C(regressed,regressed),...
                                            XX(t,~regressed)'*XX(t,~regressed))*B_S);
                                    end
                                end
                            end
                    end
                end
                avLL = avLL + Gamma(:,k).*(dist - NormWishtrace);
            end
            if hmm.train.acrosstrial_constrained, avLL = avLL / N; end
            
        elseif strcmp(distname,'bernoulli')
            L = zeros(sum(T),K);
            for k=1:K
                % expectation of log(p) is psi(a) - psi(a+b):
                pterm = X .* repmat( psi(hmm.state(k).W.a) - ...
                    psi(hmm.state(k).W.a + hmm.state(k).W.b) ,sum(T),1);
                qterm = (~X) .* repmat(psi(hmm.state(k).W.b) - ...
                    psi(hmm.state(k).W.a + hmm.state(k).W.b) ,sum(T),1);
                L(:,k) = sum(pterm + qterm,2) .* Gamma(:,k);
            end
            
            avLL = sum(L,2);
            
        elseif strcmp(distname,'poisson')
            L = zeros(sum(T),K);
            constterm = -gammaln(X+1);
            for k=1:K
                % note the expectation of log(lambda) is -log(lambda_b) + psigamma(lambda_a)
                num = (X.*(repmat(psi(hmm.state(k).W.W_shape),sum(T),1) - ...
                    log(hmm.state(k).W.W_rate))) - hmm.state(k).W.W_mean;
                num = num.*repmat(Gamma(:,k),1,ndim);
                L(:,k) = sum(num + constterm,2);
            end
            
            avLL = sum(L,2);
            
        elseif strcmp(distname,'logistic')
            if isfield(hmm,'Gamma');hmm=rmfield(hmm,'Gamma');end
            hmm.Gamma = Gamma;
            
            exp_H_LL = loglikelihoodofH(Y,X,hmm);
            avLL = exp_H_LL;
            
        end
        savLL = sum(avLL);
    end
    
end

if isempty(scale)
    FrEn=[-Entr -savLL -avLLGamma +KLdivTran +KLdiv];
else
    FrEn=[sum(-log(scale)) +KLdivTran +KLdiv];
end

% fprintf(' %.10g + %.10g + %.10g + %.10g + %.10g =  %.10g \n',...
%     sum(-Entr) ,sum(-savLL) ,sum(-avLLGamma) ,sum(+KLdivTran) ,sum(+KLdiv) ,sum(FrEn));

end

function [WishTrace,X_coded_vals] = computeWishTrace(hmm,regressed,XX,C,k)
X = XX(:,~regressed);
if length(unique(X))<5 && ~isempty(unique(X))>0
    % regressors are low dim categorical - compute and store in cache for
    % each regressor type - convert to binary code:
    X_coded = X*[2.^(1:size(X,2))]';
    X_coded_vals = unique(X_coded);
    validentries = logical(hmm.train.S(:)==1);
    WishTrace = zeros(1,length(X_coded_vals));
    %for k = 1:hmm.K
    B_S = hmm.state(k).W.S_W(validentries,validentries);
    for i=1:length(X_coded_vals)
        t_samp = find(X_coded==X_coded_vals(i),1);
        WishTrace(i) = trace(kron(C(regressed,regressed),X(t_samp,:)'*X(t_samp,:))*B_S);
    end
    %end
    
else
    WishTrace =[];
    X_coded_vals=[];
end
end