function B = obslike (X,hmm,residuals)
%
% Evaluate likelihood of data given observation model
%
% INPUT
% X          N by ndim data matrix
% hmm        hmm data structure
% residuals  in case we train on residuals, the value of those.
%
% OUTPUT
% B          Likelihood of N data points
%
% Author: Diego Vidaurre, OHBA, University of Oxford
 
[T,ndim]=size(X);
K=hmm.K;
setxx; % build XX and get orders

Tres = T-hmm.train.maxorder;
S = hmm.train.S==1; regressed = sum(S,1)>0;
ltpi = sum(regressed)/2 * log(2*pi);
B =zeros(T,K);  

%aux = zeros(T,K,4);

switch hmm.train.covtype,
    case 'uniquediag'
        ldetWishB=0;
        PsiWish_alphasum=0;
        for n=1:ndim,
            if ~regressed(n), continue; end
            ldetWishB=ldetWishB+0.5*log(hmm.Omega.Gam_rate(n));
            PsiWish_alphasum=PsiWish_alphasum+0.5*psi(hmm.Omega.Gam_shape);
        end;
        C = hmm.Omega.Gam_shape ./ hmm.Omega.Gam_rate;
    case 'uniquefull'
        ldetWishB=0.5*logdet(hmm.Omega.Gam_rate(regressed,regressed));
        PsiWish_alphasum=0;
        for n=1:sum(regressed),
            PsiWish_alphasum=PsiWish_alphasum+psi(hmm.Omega.Gam_shape/2+0.5-n/2); 
        end;
        PsiWish_alphasum=PsiWish_alphasum*0.5;
        C = hmm.Omega.Gam_shape * hmm.Omega.Gam_irate;
end;

for k=1:K
    hs=hmm.state(k);
    setstateoptions;
    
    switch train.covtype,
        case 'diag'
            ldetWishB=0;
            PsiWish_alphasum=0;
            for n=1:ndim,
                if ~regressed(n), continue; end
                ldetWishB=ldetWishB+0.5*log(hs.Omega.Gam_rate(n));
                PsiWish_alphasum=PsiWish_alphasum+0.5*psi(hs.Omega.Gam_shape);
            end;
            C = hs.Omega.Gam_shape ./ hs.Omega.Gam_rate;
        case 'full'
            ldetWishB=0.5*logdet(hs.Omega.Gam_rate(regressed,regressed));
            PsiWish_alphasum=0;
            for n=1:sum(regressed),
                PsiWish_alphasum=PsiWish_alphasum+0.5*psi(hs.Omega.Gam_shape/2+0.5-n/2);  
            end;
            C = hs.Omega.Gam_shape * hs.Omega.Gam_irate;
    end;
    
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
    
    B(hmm.train.maxorder+1:T,k)= - ltpi - ldetWishB + PsiWish_alphasum + dist - NormWishtrace; 
end
B=exp(B);
end
