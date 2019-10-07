function [FrEn,avLL] = evalfreeenergy(X,T,Gamma,Xi,hmm,residuals,XX,todo)
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

mixture_model = isfield(hmm.train,'id_mixture') && hmm.train.id_mixture;

K = length(hmm.state);
if (nargin<7 || isempty(XX)) && todo(2)==1
    setxx; % build XX and get orders
end
if isfield(hmm.state(1),'W')
    ndim = size(hmm.state(1).W.Mu_W,2);
else
    ndim = size(hmm.state(1).Omega.Gam_rate,2);
end
if isfield(hmm.train,'B'), Q = size(hmm.train.B,2);
else, Q = ndim; end
pcapred = hmm.train.pcapred>0;
if pcapred, M = hmm.train.pcapred; end

Tres = sum(T) - length(T)*hmm.train.maxorder;
S = hmm.train.S==1;
regressed = sum(S,1)>0;
ltpi = sum(regressed)/2 * log(2*pi);

if (nargin<6 || isempty(residuals)) && todo(2)==1
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
    Entr = GammaEntropy(Gamma,Xi,T,hmm.train.maxorder);
end

% loglikelihood of Gamma
avLLGamma = [];
if todo(3)==1
    avLLGamma = GammaavLL(hmm,Gamma,Xi,T);
end

% P and Pi KL
KLdivTran = [];
if todo(4)==1
    KLdivTran = KLtransition(hmm);
end

% state KL
KLdiv = [];
if todo(5)==1
    if strcmp(hmm.train.covtype,'uniquediag')
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
    for k = 1:K
        hs=hmm.state(k);
        pr=hmm.state(k).prior;
        setstateoptions;
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
                
            elseif strcmp(train.covtype,'diag') || strcmp(train.covtype,'uniquediag')
                for n=1:ndim
                    if ~regressed(n), continue; end
                    prior_prec = [];
                    if train.zeromean==0
                        prior_prec = hs.prior.Mean.iS(n);
                    end
                    if ~isempty(orders)
                        if pcapred
                            prior_prec = [prior_prec; (hs.beta.Gam_shape(:,n) ./ hs.beta.Gam_rate(:,n))];  
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
            else % full or uniquefull
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
            end
        end
        
        OmegaKL = [];
        switch train.covtype
            case 'diag'
                OmegaKL = 0;
                for n=1:ndim
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
        
        if pcapred
            betaKL = 0; 
            for n1=1:M
                for n2=1:ndim
                    betaKL = betaKL + gamma_kl(hs.beta.Gam_shape(n1,n2),pr.beta.Gam_shape(n1,n2), ...
                        hs.beta.Gam_rate(n1,n2),pr.beta.Gam_rate(n1,n2));
                end
            end
            KLdiv = [KLdiv OmegaKL betaKL WKL];
        else
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
            KLdiv = [KLdiv OmegaKL sigmaKL alphaKL WKL];
        end
    end
end

% data log likelihood
avLL = []; savLL = [];
if todo(2)==1
    avLL = zeros(Tres,1);
    if strcmp(hmm.train.covtype,'uniquediag')
        ldetWishB = 0;
        PsiWish_alphasum = 0;
        for n = 1:ndim
            if ~regressed(n), continue; end
            ldetWishB=ldetWishB+0.5*log(hmm.Omega.Gam_rate(n));
            PsiWish_alphasum=PsiWish_alphasum+0.5*psi(hmm.Omega.Gam_shape);
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
    end
    for k = 1:K
        hs = hmm.state(k);
        setstateoptions;
        if strcmp(train.covtype,'diag')
            ldetWishB = 0;
            PsiWish_alphasum=0;
            for n=1:ndim
                if ~regressed(n), continue; end
                ldetWishB=ldetWishB+0.5*log(hs.Omega.Gam_rate(n));
                PsiWish_alphasum=PsiWish_alphasum+0.5*psi(hs.Omega.Gam_shape);
            end
            C = hs.Omega.Gam_shape ./ hs.Omega.Gam_rate;
            avLL = avLL + Gamma(:,k) * (-ltpi-ldetWishB+PsiWish_alphasum);
        elseif strcmp(train.covtype,'full')
            ldetWishB=0.5*logdet(hs.Omega.Gam_rate(regressed,regressed));
            PsiWish_alphasum=0;
            for n=1:sum(regressed)
                PsiWish_alphasum=PsiWish_alphasum+0.5*psi(hs.Omega.Gam_shape/2+0.5-n/2);
            end
            C = hs.Omega.Gam_shape * hs.Omega.Gam_irate;
            avLL = avLL + Gamma(:,k) * (-ltpi-ldetWishB+PsiWish_alphasum+0.5*sum(regressed)*log(2));
        end
        meand = zeros(size(XX,1),sum(regressed)); % mean distance
        if train.uniqueAR
            for n=1:ndim
                ind = n:ndim:size(XX,2);
                meand(:,n) = XX(:,ind) * hs.W.Mu_W;
            end
        elseif ~isempty(hs.W.Mu_W)
            meand = XX * hs.W.Mu_W(:,regressed);
        end
        d = residuals(:,regressed) - meand;
        if strcmp(train.covtype,'diag') || strcmp(train.covtype,'uniquediag')
            Cd =  repmat(C(regressed)',1,Tres) .* d';
        else
            Cd = C(regressed,regressed) * d';
        end
        dist=zeros(Tres,1);
        for n=1:sum(regressed)
            dist=dist-0.5*d(:,n).*Cd(n,:)';
        end
        NormWishtrace=zeros(Tres,1); % covariance of the distance
        if ~isempty(hs.W.Mu_W)
            switch train.covtype
                case {'diag','uniquediag'}
                    for n=1:ndim
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
                                sum( (XX(:,Sind(:,n)) * permute(hs.W.S_W(n,Sind(:,n),Sind(:,n)),[2 3 1])) ...
                                .* XX(:,Sind(:,n)), 2);
                        end
                    end
                    
                case {'full','uniquefull'}
                    if isempty(orders)
                        NormWishtrace = 0.5 * sum(sum(C .* hs.W.S_W));
                    else
                        if hmm.train.pcapred>0
                            I = (0:hmm.train.pcapred+(~train.zeromean)-1) * ndim;
                        else
                            I = (0:length(orders)*Q+(~train.zeromean)-1) * ndim;
                        end
                        for n1=1:ndim
                            if ~regressed(n1), continue; end
                            index1 = I + n1; index1 = index1(Sind(:,n1));
                            tmp = (XX(:,Sind(:,n1)) * hs.W.S_W(index1,:));
                            for n2=1:ndim
                                if ~regressed(n2), continue; end
                                index2 = I + n2; index2 = index2(Sind(:,n2));
                                NormWishtrace = NormWishtrace + 0.5 * C(n1,n2) * ...
                                    sum( tmp(:,index2) .* XX(:,Sind(:,n2)),2);
                            end
                        end
                    end
            end
        end
        avLL = avLL + Gamma(:,k).*(dist - NormWishtrace);
    end
    savLL = sum(avLL);
end

FrEn=[-Entr -savLL -avLLGamma +KLdivTran +KLdiv];

end