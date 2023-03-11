function [Gamma,Gammasum,Xi,LL] = hsinference_batched(T,hmm,residuals,XX)


N = length(T);
K = hmm.K;

setstateoptions;
rangeK = 1:K;
ndim = size(residuals,2);

% Cache shared results for use in obslike
for k = rangeK
    hmm.cache.train{k} = train;
    if k == 1 && strcmp(train.covtype,'uniquediag') || strcmp(train.covtype,'shareddiag') 
        ldetWishB = 0;
        PsiWish_alphasum = 0;
        for n = 1:ndim
            if ~regressed(n), continue; end
            ldetWishB = ldetWishB+0.5*log(hmm.Omega.Gam_rate(n));
            PsiWish_alphasum = PsiWish_alphasum+0.5*psi(hmm.Omega.Gam_shape);
        end
        C = hmm.Omega.Gam_shape ./ hmm.Omega.Gam_rate;
    elseif k == 1 && strcmp(train.covtype,'uniquefull') || strcmp(train.covtype,'sharedfull')
        ldetWishB = 0.5*logdet(hmm.Omega.Gam_rate(regressed,regressed));
        PsiWish_alphasum = 0;
        for n = 1:sum(regressed)
            PsiWish_alphasum = PsiWish_alphasum+psi(hmm.Omega.Gam_shape/2+0.5-n/2);
        end
        PsiWish_alphasum=PsiWish_alphasum*0.5;
        C = hmm.Omega.Gam_shape * hmm.Omega.Gam_irate;
        for kk = rangeK
            [hmm.cache.WishTrace{kk},hmm.cache.codevals] = computeWishTrace(hmm,regressed,XX,C,kk);
        end
    elseif strcmp(train.covtype,'diag')
        ldetWishB=0;
        PsiWish_alphasum = 0;
        for n = 1:ndim
            if ~regressed(n), continue; end
            ldetWishB = ldetWishB+0.5*log(hmm.state(k).Omega.Gam_rate(n));
            PsiWish_alphasum = PsiWish_alphasum+0.5*psi(hmm.state(k).Omega.Gam_shape);
        end
        C = hmm.state(k).Omega.Gam_shape ./ hmm.state(k).Omega.Gam_rate;
        if episodic, iC = hmm.state(k).Omega.Gam_rate / hmm.state(k).Omega.Gam_shape; end
    elseif strcmp(train.covtype,'full')
        ldetWishB = 0.5*logdet(hmm.state(k).Omega.Gam_rate(regressed,regressed));
        PsiWish_alphasum = 0;
        for n = 1:sum(regressed)
            PsiWish_alphasum = PsiWish_alphasum+0.5*psi(hmm.state(k).Omega.Gam_shape/2+0.5-n/2);
        end
        C = hmm.state(k).Omega.Gam_shape * hmm.state(k).Omega.Gam_irate;
        if episodic, iC = hmm.state(k).Omega.Gam_rate / hmm.state(k).Omega.Gam_shape; end
        [hmm.cache.WishTrace{k},hmm.cache.codevals] = computeWishTrace(hmm,regressed,XX,C,k);
    end
    % Set up cache
    if ~isfield(train,'distribution') || strcmp(train.distribution,'Gaussian')
        hmm.cache.ldetWishB{k} = ldetWishB;
        hmm.cache.PsiWish_alphasum{k} = PsiWish_alphasum;
        hmm.cache.C{k} = C;
    end
end

Gamma = cell(N,1);
LL = cell(N,1);
Xi = cell(N,1);
    
for j = 1:N
    
    t0 = sum(T(1:j-1)) - order*(j-1); 
    if order > 0
        R = [zeros(order,size(residuals,2)); residuals(t0+1:t0+T(j)-order,:)];
    else
        R = residuals(t0+1:t0+T(j),:);
    end
    
    Tj = size(R,1);
    
    L = obslike([],hmm,residuals,XX,hmm.cache);
    L(L<realmin) = realmin;
    L(L>realmax) = realmax;
    L = embeddata_batched(L,Tj,hmm.train.embeddedlags_batched); 
    L = permute(mean(L,2),[1,3,2]);

    [Gamma{j},Xij,LL{j}] = fb_Gamma_inference_sub(L,hmm.P,hmm.Pi,size(L,1),order);
    
    Xi{j} = reshape(Xij,size(Xij,1),K,K);
    
end

Gamma = cell2mat(Gamma);
Xi = cell2mat(Xi);
LL = cell2mat(LL);
Gammasum = sum(Gamma);


end