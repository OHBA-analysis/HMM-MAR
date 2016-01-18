function FrEn = evalfreeenergy (X,T,Gamma,Xi,hmm,residuals)
% Computes the Free Energy of an HMM depending on observation model
%
% INPUT
% X            observations
% T            length of series
% Gamma        probability of states conditioned on data
% Xi           joint probability of past and future states conditioned on data
% hmm          data structure
% residuals    in case we train on residuals, the value of those.
%
% OUTPUT
% FrEn         value of the variational free energy, separated in the
%               different terms
%
% Author: Diego Vidaurre, OHBA, University of Oxford

ndim = size(X,2); 
K=hmm.K;
setxx; % build XX and get orders

Tres = sum(T) - length(T)*hmm.train.maxorder;
S = hmm.train.S==1;
regressed = sum(S,1)>0;

if nargin<6 || isempty(residuals)
    if ~isfield(hmm.train,'Sind'), 
        orders = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
        hmm.train.Sind = formindexes(orders,hmm.train.S); 
    end
    if ~hmm.train.zeromean, hmm.train.Sind = [true(1,ndim); hmm.train.Sind]; end
    residuals =  getresiduals(X,T,hmm.train.Sind,hmm.train.maxorder,hmm.train.order,...
        hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag,hmm.train.zeromean);
end

Gammasum=sum(Gamma,1);
ltpi = sum(regressed)/2 * log(2*pi);

Entr = 0;
for tr=1:length(T);
    t = sum(T(1:tr-1)) - (tr-1)*order + 1;
    Gamma_nz = Gamma(t,:); Gamma_nz(Gamma_nz==0) = realmin;
    Entr = Entr - sum((Gamma_nz).*log(Gamma_nz));
    t = (sum(T(1:tr-1)) - (tr-1)*(order+1) + 1) : ((sum(T(1:tr)) - tr*(order+1)));
    Xi_nz = Xi(t,:,:); Xi_nz(Xi_nz==0) = realmin;
    Psi=zeros(size(Xi_nz));                    % P(S_t|S_t-1)
    for k = 1:K,
        sXi = sum(Xi_nz(:,:,k),2);
        Psi(:,:,k) = Xi_nz(:,:,k)./repmat(sXi,1,K);
    end;
    Psi(Psi==0) = realmin;
    Entr = Entr - sum(Xi_nz(:).*log(Psi(:)));    % entropy of hidden states
end

% Free energy terms for model not including obs. model
% avLL for hidden state parameters and KL-divergence
KLdiv=dirichlet_kl(hmm.Dir_alpha,hmm.prior.Dir_alpha);
PsiDir_alphasum=psi(sum(hmm.Dir_alpha,2));

avLLGamma = 0;  
jj = zeros(length(T),1); % reference to first time point of the segments
for in=1:length(T);
    jj(in) = sum(T(1:in-1)) - hmm.train.maxorder*(in-1) + 1;
end
for l=1:K,
    % KL-divergence for transition prob
    KLdiv=[KLdiv dirichlet_kl(hmm.Dir2d_alpha(l,:),hmm.prior.Dir2d_alpha(l,:))];
    % avLL initial state  
    avLLGamma = avLLGamma + sum(Gamma(jj,l)) * (psi(hmm.Dir_alpha(l)) - PsiDir_alphasum);
end     
% avLL remaining time points  
PsiDir2d_alphasum=psi(sum(hmm.Dir2d_alpha(:)));
for k=1:K,
    for l=1:K,
        avLLGamma = avLLGamma + sum(Xi(:,l,k)) * (psi(hmm.Dir2d_alpha(l,k))-PsiDir2d_alphasum);
    end
end

OmegaKL = 0;
avLL = 0;  

if strcmp(hmm.train.covtype,'uniquediag')
    OmegaKL = 0;
    for n=1:ndim
        if ~regressed(n), continue; end
        OmegaKL = OmegaKL + gamma_kl(hmm.Omega.Gam_shape,hmm.prior.Omega.Gam_shape, ...
            hmm.Omega.Gam_rate(n),hmm.prior.Omega.Gam_rate(n));
    end;
    KLdiv=[KLdiv OmegaKL];
    ldetWishB=0;
    PsiWish_alphasum=0;
    for n=1:ndim,
        if ~regressed(n), continue; end
        ldetWishB=ldetWishB+0.5*log(hmm.Omega.Gam_rate(n));
        PsiWish_alphasum=PsiWish_alphasum+0.5*psi(hmm.Omega.Gam_shape);
    end;
    C = hmm.Omega.Gam_shape ./ hmm.Omega.Gam_rate;
    avLL=avLL+ Tres*(-ltpi-ldetWishB+PsiWish_alphasum);
elseif strcmp(hmm.train.covtype,'uniquefull')
    OmegaKL = wishart_kl(hmm.Omega.Gam_rate(regressed,regressed),hmm.prior.Omega.Gam_rate(regressed,regressed), ...
        hmm.Omega.Gam_shape,hmm.prior.Omega.Gam_shape);
    KLdiv=[KLdiv OmegaKL];
    ldetWishB=0.5*logdet(hmm.Omega.Gam_rate(regressed,regressed));
    PsiWish_alphasum=0;
    for n=1:sum(regressed),
        PsiWish_alphasum=PsiWish_alphasum+0.5*psi(hmm.Omega.Gam_shape/2+0.5-n/2);   
    end;
    C = hmm.Omega.Gam_shape * hmm.Omega.Gam_irate;
    avLL=avLL+ Tres*(-ltpi-ldetWishB+PsiWish_alphasum);
end

for k=1:K,
    hs=hmm.state(k);		 
    pr=hmm.state(k).prior;
    setstateoptions;
    
    if strcmp(train.covtype,'diag')
        ldetWishB=0;
        PsiWish_alphasum=0;
        for n=1:ndim,
            if ~regressed(n), continue; end
            ldetWishB=ldetWishB+0.5*log(hs.Omega.Gam_rate(n));
            PsiWish_alphasum=PsiWish_alphasum+0.5*psi(hs.Omega.Gam_shape);
        end;
        C = hs.Omega.Gam_shape ./ hs.Omega.Gam_rate;
        avLL = avLL+ Gammasum(k)*(-ltpi-ldetWishB+PsiWish_alphasum);
    elseif strcmp(train.covtype,'full')
        ldetWishB=0.5*logdet(hs.Omega.Gam_rate(regressed,regressed));
        PsiWish_alphasum=0;
        for n=1:sum(regressed),
            PsiWish_alphasum=PsiWish_alphasum+0.5*psi(hs.Omega.Gam_shape/2+0.5-n/2);
        end;
        C = hs.Omega.Gam_shape * hs.Omega.Gam_irate;
        avLL = avLL + Gammasum(k) * (-ltpi-ldetWishB+PsiWish_alphasum);
    end
    
    meand = zeros(size(XX{kk},1),sum(regressed));
    if train.uniqueAR
        for n=1:ndim
            ind = n:ndim:size(XX{kk},2);
            meand(:,n) = XX{kk}(:,ind) * hs.W.Mu_W;
        end
    elseif ~isempty(hs.W.Mu_W)
        meand = XX{kk} * hs.W.Mu_W(:,regressed);
    end
    d = residuals(:,regressed) - meand;
    if strcmp(train.covtype,'diag') || strcmp(train.covtype,'uniquediag')
        Cd =  repmat(C(regressed)',1,Tres) .* d';
    else
        Cd = C(regressed,regressed) * d';
    end
    
    dist=zeros(Tres,1);
    for n=1:sum(regressed),
        dist=dist-0.5*d(:,n).*Cd(n,:)';
    end
    
    NormWishtrace=zeros(Tres,1);
    if ~isempty(hs.W.Mu_W)
        switch train.covtype,
            case {'diag','uniquediag'}
                for n=1:ndim,
                    if ~regressed(n), continue; end
                    if train.uniqueAR
                        ind = n:ndim:size(XX{kk},2);
                        NormWishtrace = NormWishtrace + 0.5 * C(n) * ...
                            sum( (XX{kk}(:,ind) * hs.W.S_W) .* XX{kk}(:,ind), 2);
                    elseif ndim==1
                        NormWishtrace = NormWishtrace + 0.5 * C(n) * ...
                            sum( (XX{kk}(:,Sind(:,n)) * hs.W.S_W) ...
                            .* XX{kk}(:,Sind(:,n)), 2);                        
                    else
                        NormWishtrace = NormWishtrace + 0.5 * C(n) * ...
                            sum( (XX{kk}(:,Sind(:,n)) * permute(hs.W.S_W(n,Sind(:,n),Sind(:,n)),[2 3 1])) ...
                            .* XX{kk}(:,Sind(:,n)), 2);
                    end
                end;
                
            case {'full','uniquefull'}
                if isempty(orders)
                    NormWishtrace = 0.5 * sum(sum(C .* hs.W.S_W));
                else
                    I = (0:length(orders)*ndim+(~train.zeromean)-1) * ndim;
                    for n1=1:ndim
                        if ~regressed(n1), continue; end
                        index1 = I + n1; index1 = index1(Sind(:,n1)); 
                        tmp = (XX{kk}(:,Sind(:,n1)) * hs.W.S_W(index1,:));
                        for n2=1:ndim
                            if ~regressed(n2), continue; end
                            index2 = I + n2; index2 = index2(Sind(:,n2));
                            NormWishtrace = NormWishtrace + 0.5 * C(n1,n2) * ...
                                sum( tmp(:,index2) .* XX{kk}(:,Sind(:,n2)),2);
                        end
                    end
                end
        end
    end
    
    avLL = avLL + sum(Gamma(:,k).*(dist - NormWishtrace));
    
    WKL = 0;
    if ~isempty(hs.W.Mu_W) 
        if train.uniqueAR || ndim==1
            if ~isempty(train.prior)
                prior_var = train.prior.S;
                prior_mu = train.prior.Mu;
            else
                prior_prec = [];
                if train.zeromean==0
                    prior_prec = hs.prior.Mean.iS(n);
                end
                prior_prec = [prior_prec (hs.alpha.Gam_shape ./  hs.alpha.Gam_rate)];
                prior_var = diag(1 ./ prior_prec);
                prior_mu = zeros(length(orders) + ~train.zeromean,1);
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
                    ndim_n = sum(S(:,n));
                    alphamat = repmat( (hs.alpha.Gam_shape ./  hs.alpha.Gam_rate), ndim_n, 1);
                    prior_prec = [prior_prec; repmat(hs.sigma.Gam_shape(S(:,n)==1,n) ./ ...
                        hs.sigma.Gam_rate(S(:,n)==1,n), length(orders), 1) .* alphamat(:)] ;
                end
                prior_var = diag(1 ./ prior_prec);
                WKL = WKL + gauss_kl(hs.W.Mu_W(Sind(:,n),n),zeros(sum(Sind(:,n)),1), ...
                    permute(hs.W.S_W(n,Sind(:,n),Sind(:,n)),[2 3 1]), prior_var);
            end;
        else % full or uniquefull
            prior_prec = [];
            if train.zeromean==0
                prior_prec = hs.prior.Mean.iS;
            end
            if ~isempty(orders)
                sigmaterm = (hs.sigma.Gam_shape(S==1) ./ hs.sigma.Gam_rate(S==1) );
                sigmaterm = repmat(sigmaterm, length(orders), 1);
                alphaterm = repmat( (hs.alpha.Gam_shape ./  hs.alpha.Gam_rate), sum(S(:)), 1);
                alphaterm = alphaterm(:);
                prior_prec = [prior_prec; alphaterm .* sigmaterm];
            end
            prior_var = diag(1 ./ prior_prec);
            mu_w = hs.W.Mu_W';
            mu_w = mu_w(Sind);
            WKL = gauss_kl(mu_w,zeros(length(mu_w),1), hs.W.S_W, prior_var);
        end
    end
    
    switch train.covtype
        case 'diag'
            OmegaKL = 0;
            for n=1:ndim
                if ~regressed(n), continue; end 
                OmegaKL = OmegaKL + gamma_kl(hs.Omega.Gam_shape,hs.prior.Omega.Gam_shape, ...
                    hs.Omega.Gam_rate(n),hs.prior.Omega.Gam_rate(n));
            end;
        case 'full'
            OmegaKL = wishart_kl(hs.Omega.Gam_rate(regressed,regressed),hs.prior.Omega.Gam_rate(regressed,regressed), ...
                hs.Omega.Gam_shape,hs.prior.Omega.Gam_shape);
    end
    
    sigmaKL = 0;
    if isempty(train.prior) && ~isempty(orders) && ~train.uniqueAR && ndim>1
        for n1=1:ndim
            for n2=1:ndim
                if (train.symmetricprior && n2<n1) || S(n1,n2)==0, continue; end
                sigmaKL = sigmaKL + gamma_kl(hs.sigma.Gam_shape(n1,n2),pr.sigma.Gam_shape(n1,n2), ...
                    hs.sigma.Gam_rate(n1,n2),pr.sigma.Gam_rate(n1,n2));
            end;
        end;
    end
    
    alphaKL = 0;
    if isempty(train.prior) &&  ~isempty(orders)
        for i=1:length(orders)
            alphaKL = alphaKL + gamma_kl(hs.alpha.Gam_shape,pr.alpha.Gam_shape, ...
                hs.alpha.Gam_rate(i),pr.alpha.Gam_rate(i));
        end
    end
    
    KLdiv=[KLdiv OmegaKL sigmaKL alphaKL WKL];
    
    %if any(isnan([OmegaKL sigmaKL alphaKL WKL])), keyboard; end
    
end;

FrEn=[-Entr -avLL -avLLGamma +KLdiv];
[sum(-Entr) sum(-avLL) sum(-avLLGamma) sum(KLdiv) sum(FrEn)]