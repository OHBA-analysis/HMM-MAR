function [FrEn,avLL] = evalfreeenergy_addHMM(X,T,Gamma,Xi,hmm,residuals,XX,todo)
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
%                   elements 5-: KL for the state parameters
% avLL         log likelihood of the observed data, per trial
%
% Author: Diego Vidaurre, OHBA, University of Oxford

if nargin<8, todo = ones(1,5); end

p = hmm.train.lowrank; do_HMM_pca = (p > 0);
K = hmm.K;
pcapred = hmm.train.pcapred>0;

if (nargin<7 || isempty(XX)) && todo(2)==1
    setxx; % build XX and get orders
end
if isfield(hmm.state(1),'W')
    if do_HMM_pca
        ndim = size(hmm.state(1).W.Mu_W,1);
    else
        ndim = size(hmm.state(1).W.Mu_W,2);
    end
else
    ndim = size(hmm.state(1).Omega.Gam_rate,2);
end

Tres = sum(T) - length(T)*hmm.train.maxorder;
S = hmm.train.S==1;
setstateoptions;
regressed = sum(S,1)>0;
ltpi = sum(regressed)/2 * log(2*pi);


if sum(~regressed)>1 && ~isempty(X)
    %catch special case where regressors do not vary within trials:
    for t=1:length(T)
        varcheck(t,:) = var(X(sum(T(1:t-1))+[1:T(t)-1],~regressed));
    end
    trialwiseregression = all(varcheck(:)==0);
else
    trialwiseregression = false;
end

if ~do_HMM_pca && (nargin<6 || isempty(residuals)) && todo(2)==1
    if ~isfield(hmm.train,'Sind')
        orders = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
        hmm.train.Sind = formindexes(orders,hmm.train.S);
    end
    if ~hmm.train.zeromean, hmm.train.Sind = [true(1,ndim); hmm.train.Sind]; end
    residuals =  getresiduals(X,T,hmm.train.Sind,hmm.train.maxorder,hmm.train.order,...
        hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag,hmm.train.zeromean);
end

% Entropy of Gamma
Entr = [];
if todo(1)==1
    Entr = 0;
    for k = 1:K
        Gammak = [Gamma(:,k) (1-Gamma(:,k))];
        Xik = permute(Xi(:,k,:,:),[1 3 4 2]);
        Entr = Entr + GammaEntropy(Gammak,Xik,T,hmm.train.maxorder);
    end
end

% loglikelihood of Gamma
avLLGamma = [];
if todo(3)==1
    avLLGamma = GammaavLL_addHMM(hmm,Xi);
end

% P and Pi KL
KLdivTran = [];
if todo(4)==1
    KLdivTran = KLtransition_addHMM(hmm);
end

% state KL
KLdiv = [];
if todo(5)==1
    
    if do_HMM_pca
        OmegaKL = gamma_kl(hmm.Omega.Gam_shape,hmm.prior.Omega.Gam_shape, ...
            hmm.Omega.Gam_rate,hmm.prior.Omega.Gam_rate);
        KLdiv = [KLdiv OmegaKL];
    elseif strcmp(hmm.train.covtype,'uniquediag')
        OmegaKL = 0;
        for n = 1:ndim
            if ~regressed(n), continue; end
            OmegaKL = OmegaKL + gamma_kl(hmm.Omega.Gam_shape,hmm.prior.Omega.Gam_shape, ...
                hmm.Omega.Gam_rate(n),hmm.prior.Omega.Gam_rate(n));
        end
        KLdiv = [KLdiv OmegaKL];
    elseif strcmp(hmm.train.covtype,'uniquefull')
        OmegaKL = wishart_kl(hmm.Omega.Gam_rate(regressed,regressed),...
            hmm.prior.Omega.Gam_rate(regressed,regressed), ...
            hmm.Omega.Gam_shape,hmm.prior.Omega.Gam_shape);
        KLdiv = [KLdiv OmegaKL];
    end
    
    for k = 1:K % we don't go until K+1 because it doesn't change after initialisation
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
                
            elseif strcmp(train.covtype,'diag') || strcmp(train.covtype,'uniquediag')
                for n = 1:ndim
                    if ~regressed(n), continue; end
                    prior_prec = [];
                    if train.zeromean==0
                        prior_prec = hs.prior.Mean.iS(n);
                    end
                    if ~isempty(orders)
                        ndim_n = sum(S(:,n));
                        alphamat = repmat( (hs.alpha.Gam_shape ./  hs.alpha.Gam_rate), ndim_n, 1);
                        prior_prec = [prior_prec; repmat(hs.sigma.Gam_shape(S(:,n)==1,n) ./ ...
                            hs.sigma.Gam_rate(S(:,n)==1,n), length(orders), 1) .* alphamat(:)] ;
                    end
                    prior_var = diag(1 ./ prior_prec);
                    WKL = WKL + gauss_kl(hs.W.Mu_W(Sind(:,n),n),zeros(sum(Sind(:,n)),1), ...
                        permute(hs.W.S_W(n,Sind(:,n),Sind(:,n)),[2 3 1]), prior_var);
                end
                
            else % full or uniquefull
                if all(S(:)==1)
                    prior_prec = [];
                    if train.zeromean==0
                        prior_prec = hs.prior.Mean.iS;
                    end
                    if ~isempty(orders)
                        sigmaterm = (hs.sigma.Gam_shape ./ hs.sigma.Gam_rate)';
                        sigmaterm = repmat(sigmaterm(:), length(orders), 1);
                        alphaterm = repmat( (hs.alpha.Gam_shape ./  hs.alpha.Gam_rate), ndim*Q, 1);
                        alphaterm = alphaterm(:);
                        prior_prec = [prior_prec; alphaterm .* sigmaterm];
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
                    sigmaterm = (hs.sigma.Gam_shape(regressed,regressed) ./ hs.sigma.Gam_rate(regressed,regressed))';
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

% data log likelihood
savLL = [];
if todo(2)==1
    
    avLL = zeros(Tres,1);
    Gamma = [Gamma (K-sum(Gamma,2))];
    any_W = hmm.train.order > 0 || hmm.train.zeromean == 0; 

    if strcmp(hmm.train.covtype,'uniquediag')
        ldetWishB = 0;
        PsiWish_alphasum = 0;
        for n = 1:ndim
            if ~regressed(n), continue; end
            ldetWishB = ldetWishB+0.5*log(hmm.Omega.Gam_rate(n));
            PsiWish_alphasum = PsiWish_alphasum+0.5*psi(hmm.Omega.Gam_shape);
        end
        C = hmm.Omega.Gam_shape ./ hmm.Omega.Gam_rate;
        avLL = avLL + (-ltpi-ldetWishB+PsiWish_alphasum);
    elseif strcmp(hmm.train.covtype,'uniquefull')
        ldetWishB = 0.5*logdet(hmm.Omega.Gam_rate(regressed,regressed));
        PsiWish_alphasum = 0;
        for n = 1:sum(regressed)
            PsiWish_alphasum=PsiWish_alphasum+0.5*psi(hmm.Omega.Gam_shape/2+0.5-n/2);
        end
        C = hmm.Omega.Gam_shape * hmm.Omega.Gam_irate;
        avLL = avLL + (-ltpi-ldetWishB+PsiWish_alphasum+0.5*sum(regressed)*log(2));
    else
        error('Not implemented beyond here')
    end
 
    % compute the mean response
    meand = zeros(size(XX,1),sum(regressed)); % mean distance
    if any_W
        for k = 1:K+1
            hs = hmm.state(k);
            if train.uniqueAR
                for n = 1:ndim
                    ind = n:ndim:size(XX,2);
                    meand(:,n) = meand(:,n) + bsxfun(@times,  XX(:,ind) * hs.W.Mu_W, Gamma(:,k));
                end
            else
                meand = meand(:,n) + bsxfun(@times, XX * hs.W.Mu_W(:,regressed), Gamma(:,k));
            end
        end
    end
    d = residuals(:,regressed) - meand;
    Cd =  repmat(C(regressed)',1,Tres) .* d';
    dist = zeros(Tres,1);
    for n = 1:sum(regressed)
        dist = dist - 0.5 * (d(:,n).*Cd(n,:)');
    end
    
    % Covariance of the distance
    NormWishtrace = zeros(Tres,1); 
    if any_W
        for k = 1:K+1 % this can probably be optimised for efficiency, summing all C's at once
            NormWishtrace_k = zeros(Tres,1);
            if strcmp(hmm.train.covtype,'uniquediag')
                for n = 1:ndim
                    if ~regressed(n), continue; end
                    if train.uniqueAR
                        ind = n:ndim:size(XX,2);
                        NormWishtrace_k = NormWishtrace_k + 0.5 * C(n) * ...
                            sum( (XX(:,ind) * hs.W.S_W) .* XX(:,ind), 2);
                    elseif ndim==1
                        NormWishtrace_k = NormWishtrace_k + 0.5 * C(n) * ...
                            sum( (XX(:,Sind(:,n)) * hs.W.S_W) ...
                            .* XX(:,Sind(:,n)), 2);
                    else
                        NormWishtrace_k = NormWishtrace_k + 0.5 * C(n) * ...
                            sum( (XX(:,Sind(:,n)) * ...
                            permute(hs.W.S_W(n,Sind(:,n),Sind(:,n)),[2 3 1])) ...
                            .* XX(:,Sind(:,n)), 2);
                    end
                end
            else %if strcmp(hmm.train.covtype,'uniquefull')
                if isempty(orders)
                    NormWishtrace_k = 0.5 * sum(sum(C .* hs.W.S_W));
                else
                    I = (0:length(orders)*ndim+(~train.zeromean)-1) * ndim;
                    if all(S(:)==1) || ~trialwiseregression
                        for n1 = 1:ndim
                            if ~regressed(n1), continue; end
                            index1 = I + n1; index1 = index1(Sind(:,n1));
                            tmp = (XX(:,Sind(:,n1)) * hs.W.S_W(index1,:));
                            for n2=1:ndim
                                if ~regressed(n2), continue; end
                                index2 = I + n2; index2 = index2(Sind(:,n2));
                                NormWishtrace_k = NormWishtrace_k + 0.5 * C(n1,n2) * ...
                                    sum( tmp(:,index2) .* XX(:,Sind(:,n2)),2);
                            end
                        end
                    else
                        error('Not implemented beyond here') 
                    end
                end
            end
            NormWishtrace = NormWishtrace - NormWishtrace_k .* Gamma(:,k);
        end
    end
    
    avLL = avLL + dist + NormWishtrace;
    savLL = sum(avLL);
end    

FrEn=[-Entr -savLL -avLLGamma +KLdivTran +KLdiv];

fprintf(' %.10g %.10g %.10g %.10g %.10g %.10g \n',...
    sum(-Entr) ,sum(-savLL) ,sum(-avLLGamma) ,sum(+KLdivTran) ,sum(+KLdiv) ,sum(FrEn));

end
