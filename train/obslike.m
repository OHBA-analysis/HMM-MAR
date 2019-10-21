function L = obslike(X,hmm,residuals,XX,cache)
%
% Evaluate likelihood of data given observation model, for one continuous trial
%
% INPUT
% X          N by ndim data matrix
% hmm        hmm data structure
% residuals  in case we train on residuals, the value of those.
% XX        alternatively to X (which in this case can be specified as []),
%               XX can be provided as computed by setxx.m
% OUTPUT
% B          Likelihood of N data points
%
% Author: Diego Vidaurre, OHBA, University of Oxford

if nargin < 5 || isempty(cache) 
    use_cache = false;
else
    use_cache = true;
end

K = hmm.K;
if nargin<4 || size(XX,1)==0
    [T,ndim]=size(X);
    setxx; % build XX and get orders
else
    [T,ndim] = size(residuals);
    T = T + hmm.train.maxorder;
end
if isfield(hmm.train,'B'), Q = size(hmm.train.B,2);
else Q = ndim; end

if nargin<3 || isempty(residuals)
    ndim = size(X,2);
    if ~isfield(hmm.train,'Sind')
        if hmm.train.pcapred==0
            orders = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
            hmm.train.Sind = formindexes(orders,hmm.train.S);
        else
           hmm.train.Sind = ones(hmm.train.pcapred,ndim); 
        end
    end
    if ~hmm.train.zeromean, hmm.train.Sind = [true(1,ndim); hmm.train.Sind]; end
    residuals =  getresiduals(X,T,hmm.train.Sind,hmm.train.maxorder,hmm.train.order,...
        hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag,hmm.train.zeromean);
end

Tres = T-hmm.train.maxorder;
S = hmm.train.S==1; 
regressed = sum(S,1)>0;
ltpi = sum(regressed)/2 * log(2*pi);
L = zeros(T,K);  

switch hmm.train.covtype
    case 'uniquediag'
        ldetWishB = 0;
        PsiWish_alphasum = 0;
        for n = 1:ndim
            if ~regressed(n), continue; end
            ldetWishB = ldetWishB+0.5*log(hmm.Omega.Gam_rate(n));
            PsiWish_alphasum = PsiWish_alphasum+0.5*psi(hmm.Omega.Gam_shape);
        end
        C = hmm.Omega.Gam_shape ./ hmm.Omega.Gam_rate;
    case 'uniquefull'
        ldetWishB = 0.5*logdet(hmm.Omega.Gam_rate(regressed,regressed));
        PsiWish_alphasum = 0;
        for n = 1:sum(regressed)
            PsiWish_alphasum = PsiWish_alphasum+psi(hmm.Omega.Gam_shape/2+0.5-n/2); 
        end
        PsiWish_alphasum = PsiWish_alphasum*0.5;
        C = hmm.Omega.Gam_shape * hmm.Omega.Gam_irate;
end

for k = 1:K

    if use_cache
        train = cache.train{k};
        orders = cache.orders{k};
        Sind = cache.Sind{k};
        ldetWishB = cache.ldetWishB{k};
        PsiWish_alphasum  = cache.PsiWish_alphasum{k};
        C = cache.C{k};
        do_normwishtrace = cache.do_normwishtrace;
    else
        setstateoptions;
        do_normwishtrace = ~isempty(hmm.state(k).W.Mu_W);
    
        switch train.covtype
            case 'diag'
                ldetWishB = 0;
                PsiWish_alphasum = 0;
                for n = 1:ndim
                    if ~regressed(n), continue; end
                    ldetWishB = ldetWishB+0.5*log(hmm.state(k).Omega.Gam_rate(n));
                    PsiWish_alphasum = PsiWish_alphasum+0.5*psi(hmm.state(k).Omega.Gam_shape);
                end
                C = hmm.state(k).Omega.Gam_shape ./ hmm.state(k).Omega.Gam_rate;
            case 'full'
                ldetWishB = 0.5*logdet(hmm.state(k).Omega.Gam_rate(regressed,regressed));
                PsiWish_alphasum = 0;
                for n = 1:sum(regressed)
                    PsiWish_alphasum = PsiWish_alphasum+0.5*psi(hmm.state(k).Omega.Gam_shape/2+0.5-n/2);  
                end
                C = hmm.state(k).Omega.Gam_shape * hmm.state(k).Omega.Gam_irate;
        end
    end
        
    meand = zeros(size(XX,1),sum(regressed));
    if train.uniqueAR
        for n = 1:ndim
            ind = n:ndim:size(XX,2);
            meand(:,n) = XX(:,ind) * hmm.state(k).W.Mu_W;
        end
    elseif do_normwishtrace
        meand = XX * hmm.state(k).W.Mu_W(:,regressed);
    end
    d = residuals(:,regressed) - meand;
    if strcmp(train.covtype,'diag') || strcmp(train.covtype,'uniquediag')
        Cd = bsxfun(@times,C(regressed).',d.');
    else
        Cd = C(regressed,regressed) * d';
    end
    
    dist = zeros(Tres,1);
    for n = 1:sum(regressed)
        dist = dist-0.5*d(:,n).*Cd(n,:)';
    end
    
    NormWishtrace = zeros(Tres,1);
    if do_normwishtrace
        switch train.covtype
            case {'diag','uniquediag'}
                for n = 1:ndim
                    if ~regressed(n), continue; end
                    if train.uniqueAR
                        ind = n:ndim:size(XX,2);
                        NormWishtrace = NormWishtrace + 0.5 * C(n) * ...
                            sum( (XX(:,ind) * hmm.state(k).W.S_W) .* XX(:,ind), 2);
                    elseif ndim==1
                        NormWishtrace = NormWishtrace + 0.5 * C(n) * ...
                            sum( (XX(:,Sind(:,n)) * hmm.state(k).W.S_W) ...
                            .* XX(:,Sind(:,n)), 2);
                    else
                        NormWishtrace = NormWishtrace + 0.5 * C(n) * ...
                            sum( (XX(:,Sind(:,n)) * permute(hmm.state(k).W.S_W(n,Sind(:,n),Sind(:,n)),[2 3 1])) ...
                            .* XX(:,Sind(:,n)), 2);
                    end
                end
                
            case {'full','uniquefull'}
                if isempty(orders) 
                    NormWishtrace = 0.5 * sum(sum(C .* hmm.state(k).W.S_W));
                else
                    if hmm.train.pcapred>0
                        I = (0:hmm.train.pcapred+(~train.zeromean)-1) * ndim;
                    else
                        I = (0:length(orders)*Q+(~train.zeromean)-1) * ndim;
                    end
                    for n1 = 1:ndim
                        if ~regressed(n1), continue; end
                        index1 = I + n1; index1 = index1(Sind(:,n1)); 
                        tmp = (XX(:,Sind(:,n1)) * hmm.state(k).W.S_W(index1,:));
                        for n2 = 1:ndim
                            if ~regressed(n2), continue; end
                            index2 = I + n2; index2 = index2(Sind(:,n2));
                            NormWishtrace = NormWishtrace + 0.5 * C(n1,n2) * ...
                                sum( tmp(:,index2) .* XX(:,Sind(:,n2)),2);
                        end
                    end
                end
        end
    end
    
    L(hmm.train.maxorder+1:T,k)= - ltpi - ldetWishB + PsiWish_alphasum + dist - NormWishtrace; 
    
end
L = exp(L);
end
