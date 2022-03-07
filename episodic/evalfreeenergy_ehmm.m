function [FrEn,avLL] = evalfreeenergy_ehmm(T,Gamma,Xi,ehmm,residuals,XX,todo)
% Computes the Free Energy of an HMM depending on observation model
%
% INPUT
% T            length of series
% Gamma        probability of states conditioned on data
% Xi           joint probability of past and future states conditioned on data
% ehmm          bsmm structure
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

if nargin<7, todo = ones(1,5); end

K = ehmm.K;  

if nargin>=6, Tres = size(residuals,1);
else, Tres = sum(T) - length(T)*ehmm.train.maxorder;
end
setstateoptions;
ltpi = sum(regressed)/2 * log(2*pi);
np = size(XX,2); ndim = size(residuals,2);

% Entropy of Gamma
Entr = [];
if todo(1)==1
    Entr = GammaEntropy_ehmm(Gamma,Xi,T);
end

% loglikelihood of Gamma
avLLGamma = [];
if todo(3)==1
    avLLGamma = GammaavLL_ehmm(ehmm,Xi);
end

% P and Pi KL
KLdivTran = [];
if todo(4)==1
    KLdivTran = KLtransition_ehmm(ehmm);
end

% state KL
KLdiv = [];
if todo(5)==1
    
    OmegaKL = 0;
    for n = 1:ndim
        if ~regressed(n), continue; end
        OmegaKL = OmegaKL + gamma_kl(ehmm.Omega.Gam_shape,ehmm.prior.Omega.Gam_shape, ...
            ehmm.Omega.Gam_rate(n),ehmm.prior.Omega.Gam_rate(n));
    end
    KLdiv = [KLdiv OmegaKL];
    
    WKL = 0;
    Sind_all = [repmat(Sind(:,n),K,1) == 1; false(np,1)];
    for n = 1:ndim
        prior_prec = [];
        for k = 1:K
            hs = ehmm.state(k);
            pr = ehmm.state(k).prior;
            if ~regressed(n), continue; end
            if ~train.zeromean, prior_prec = hs.prior.Mean.iS(n); end
            if train.order > 0
                ndim_n = sum(S(:,n));
                alphaterm = repmat( (hs.alpha.Gam_shape ./  hs.alpha.Gam_rate), ndim_n, 1);
                if ndim==1
                    prior_prec = [prior_prec; alphaterm(:)] ;
                else
                    sigmaterm = repmat(hs.sigma.Gam_shape(S(:,n),n) ./ ...
                        hs.sigma.Gam_rate(S(:,n),n), length(orders), 1);
                    prior_prec = [prior_prec; sigmaterm .* alphaterm(:)] ;
                end
            end
        end
        prior_var = diag(1 ./ prior_prec);
        WKL = WKL + gauss_kl(ehmm.state_shared(n).Mu_W(Sind_all),zeros(np*K,1), ...
            ehmm.state_shared(n).S_W(Sind_all,Sind_all), prior_var);
    end
    KLdiv = [KLdiv WKL];
    
    sigmaKL = 0; alphaKL = 0;
    for k = 1:K+1
        if isempty(train.prior) && ~isempty(orders) && ~train.uniqueAR && ndim>1
            for n1 = 1:ndim
                for n2 = 1:ndim
                    if (train.symmetricprior && n2<n1) || S(n1,n2)==0, continue; end
                    sigmaKL = sigmaKL + gamma_kl(hs.sigma.Gam_shape(n1,n2),pr.sigma.Gam_shape(n1,n2), ...
                        hs.sigma.Gam_rate(n1,n2),pr.sigma.Gam_rate(n1,n2));
                end
            end
        end
        if isempty(train.prior) &&  ~isempty(orders)
            for i = 1:length(orders)
                alphaKL = alphaKL + gamma_kl(hs.alpha.Gam_shape,pr.alpha.Gam_shape, ...
                    hs.alpha.Gam_rate(i),pr.alpha.Gam_rate(i));
            end
        end
    end
    KLdiv = [KLdiv sigmaKL alphaKL];
    
end

% data log likelihood
savLL = []; 
if todo(2)==1
    
    Gamma = [Gamma prod(1-Gamma,2) ];
    Gamma = rdiv(Gamma,sum(Gamma,2));
    
    avLL = zeros(Tres,1);
    
    ldetWishB = 0;
    PsiWish_alphasum = 0;
    for n = 1:ndim
        if ~regressed(n), continue; end
        ldetWishB = ldetWishB+0.5*log(ehmm.Omega.Gam_rate(n));
        PsiWish_alphasum = PsiWish_alphasum+0.5*psi(ehmm.Omega.Gam_shape);
    end
    C = ehmm.Omega.Gam_shape ./ ehmm.Omega.Gam_rate;
    avLL = avLL + (-ltpi-ldetWishB+PsiWish_alphasum);
    
    for k = 1:K+1
    
        % error
        meand = XX * ehmm.state(n).W.Mu_W(:,regressed);
        d = residuals(:,regressed) - meand;
        Cd = bsxfun(@times, C(regressed), d)';
        dist = zeros(Tres,1);
        for n = 1:sum(regressed)
            dist = dist - 0.5 * (d(:,n).*Cd(n,:)');
        end
  
        % Covariance of the distance
        NormWishtrace = zeros(Tres,1);
%         if ndim == 1
%             NormWishtrace = NormWishtrace + 0.5 * C * ...
%                 sum( (XX * ehmm.state(k).W.S_W) .* XX, 2);
%         else
%             for n = 1:ndim
%                 if ~regressed(n), continue; end
%                 NormWishtrace = NormWishtrace + 0.5 * C(n) * ...
%                     sum( (XX(:,Sind(:,n)) * ...
%                     permute(ehmm.state(k).W.S_W(n,Sind(:,n),Sind(:,n)),[2 3 1])) ...
%                     .* XX(:,Sind(:,n)), 2);
%             end
%         end
        
        avLL = avLL + Gamma(:,k) .* (dist - NormWishtrace);
    
    end    
    savLL = sum(avLL);
end

FrEn = [-Entr -savLL -avLLGamma +KLdivTran +KLdiv];

% fprintf(' %.10g + %.10g + %.10g + %.10g + %.10g + %.10g =  %.10g \n',...
%     sum(-Entr) ,sum(-(-ltpi-ldetWishB+PsiWish_alphasum)*Tres) , sum(-(dist - NormWishtrace)), ...
%     sum(-avLLGamma) , sum(+KLdivTran) ,sum(+KLdiv) ,sum(FrEn));

end
