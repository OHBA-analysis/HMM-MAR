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

K = hmm.K;

if nargin>=6, Tres = size(residuals,1); 
else, Tres = sum(T) - length(T)*hmm.train.maxorder;
end
setstateoptions;
regressed = sum(S,1)>0;
ltpi = sum(regressed)/2 * log(2*pi);
np = size(XX,2); ndim = size(residuals,2);

if sum(~regressed)>1 && ~isempty(X)
    %catch special case where regressors do not vary within trials:
    for t=1:length(T)
        varcheck(t,:) = var(X(sum(T(1:t-1))+[1:T(t)-1],~regressed));
    end
    trialwiseregression = all(varcheck(:)==0);
else
    trialwiseregression = false;
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
    
    OmegaKL = 0;
    for n = 1:ndim
        if ~regressed(n), continue; end
        OmegaKL = OmegaKL + gamma_kl(hmm.Omega.Gam_shape,hmm.prior.Omega.Gam_shape, ...
            hmm.Omega.Gam_rate(n),hmm.prior.Omega.Gam_rate(n));
    end
    KLdiv = [KLdiv OmegaKL];
    
    for k = 1:K+1
        hs = hmm.state(k);
        pr = hmm.state(k).prior;
        WKL = 0;
        
        if ~isempty(hs.W.Mu_W)
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
        end
        
        KLdiv = [KLdiv WKL];
        
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

% data log likelihood
if todo(2)==1
    
    avLL = zeros(Tres,1);
    
    ldetWishB = 0;
    PsiWish_alphasum = 0;
    for n = 1:ndim
        if ~regressed(n), continue; end
        ldetWishB = ldetWishB+0.5*log(hmm.Omega.Gam_rate(n));
        PsiWish_alphasum = PsiWish_alphasum+0.5*psi(hmm.Omega.Gam_shape);
    end
    C = hmm.Omega.Gam_shape ./ hmm.Omega.Gam_rate;
    avLL = avLL + (-ltpi-ldetWishB+PsiWish_alphasum);
    
    % compute the mean response
    meand = computeStateResponses(XX,hmm,Gamma);
    d = residuals(:,regressed) - meand;
    Cd = bsxfun(@times, C(regressed), d)'; 
    dist = zeros(Tres,1);
    for n = 1:sum(regressed)
        dist = dist - 0.5 * (d(:,n).*Cd(n,:)');
    end
        
    % Covariance of the distance
    Gamma = [Gamma (K-sum(Gamma,2)) ];
    NormWishtrace = zeros(Tres,1);
    
    X = zeros(size(XX,1),np * (K+1));
    for k = 1:K+1
        X(:,(1:np) + (k-1)*np) = bsxfun(@times, XX, Gamma(:,k));
    end
    
    for n = 1:ndim
        if ~regressed(n), continue; end
        Sind_all = [];
        for k = 1:K+1
            Sind_all = [Sind_all; Sind(:,n)];
        end
        Sind_all = Sind_all == 1; 
        if ndim==1
            NormWishtrace = NormWishtrace + 0.5 * C(n) * ...
                sum( (X(:,Sind_all) * hmm.state_shared(n).S_W(Sind_all,Sind_all)) ...
                .* X(:,Sind_all), 2);
        else
            NormWishtrace = NormWishtrace + 0.5 * C(n) * ...
                sum( (X(:,Sind_all) * ...
                hmm.state_shared(n).S_W(Sind_all,Sind_all)) ...
                .* X(:,Sind_all), 2);
        end
    end
   
    avLL = avLL + dist + NormWishtrace;
    savLL = sum(avLL);
    
    FrEn=[-Entr -savLL -avLLGamma +KLdivTran +KLdiv];
    
    %fprintf(' %.10g %.10g %.10g %.10g %.10g %.10g \n',...
    %    sum(-Entr) ,sum(-savLL) ,sum(-avLLGamma) ,sum(+KLdivTran) ,sum(+KLdiv) ,sum(FrEn));
    
end
