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
p = hmm.train.lowrank; do_HMM_pca = (p > 0);

K = hmm.K;
if nargin<4 || size(XX,1)==0
    [T,ndim] = size(X);
    setxx; % build XX and get orders
elseif ~do_HMM_pca
    [T,ndim] = size(residuals);
    T = T + hmm.train.maxorder;
else
    [T,ndim] = size(XX);
end
if isfield(hmm.train,'B'), Q = size(hmm.train.B,2);
else, Q = ndim; end

if ~do_HMM_pca && (nargin<3 || isempty(residuals))
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

if do_HMM_pca
    % This is done later because ldetWishB needs W, so it's state dependent
    ldetWishB = 0;
    PsiWish_alphasum = 0;
elseif strcmpi(hmm.train.covtype,'uniquediag')  
    ldetWishB = 0;
    PsiWish_alphasum = 0;
    for n = 1:ndim
        if ~regressed(n), continue; end
        ldetWishB = ldetWishB+0.5*log(hmm.Omega.Gam_rate(n));
        PsiWish_alphasum = PsiWish_alphasum+0.5*psi(hmm.Omega.Gam_shape);
    end
    C = hmm.Omega.Gam_shape ./ hmm.Omega.Gam_rate;
elseif strcmpi(hmm.train.covtype,'uniquefull')
    ldetWishB = 0.5*logdet(hmm.Omega.Gam_rate(regressed,regressed));
    PsiWish_alphasum = 0;
    for n = 1:sum(regressed)
        PsiWish_alphasum = PsiWish_alphasum+psi(hmm.Omega.Gam_shape/2+0.5-n/2);
    end
    PsiWish_alphasum = PsiWish_alphasum*0.5;
    C = hmm.Omega.Gam_shape * hmm.Omega.Gam_irate;
    
end

if ~(use_cache && ~do_HMM_pca)
    setstateoptions;
end

for k = 1:K

    if use_cache
        train = cache.train{k};
        orders = cache.orders{k};
        Sind = cache.Sind{k};
        ldetWishB = cache.ldetWishB{k};
        PsiWish_alphasum  = cache.PsiWish_alphasum{k};
        C = cache.C{k};
        if isfield(cache,'WishTrace')
            WishTrace = cache.WishTrace{k};
        else
            WishTrace =[];
        end
    else
        if do_HMM_pca
            W = hmm.state(k).W.Mu_W;
            v = hmm.Omega.Gam_rate / hmm.Omega.Gam_shape;
            C = W * W' + v * eye(ndim); 
            ldetWishB = 0.5*logdet(C); PsiWish_alphasum = 0;
        elseif strcmpi(hmm.train.covtype,'diag')
            ldetWishB = 0;
            PsiWish_alphasum = 0;
            for n = 1:ndim
                if ~regressed(n), continue; end
                ldetWishB = ldetWishB + 0.5*log(hmm.state(k).Omega.Gam_rate(n));
                PsiWish_alphasum = PsiWish_alphasum + 0.5*psi(hmm.state(k).Omega.Gam_shape);
            end
            C = hmm.state(k).Omega.Gam_shape ./ hmm.state(k).Omega.Gam_rate;
        elseif strcmpi(hmm.train.covtype,'full')
            ldetWishB = 0.5*logdet(hmm.state(k).Omega.Gam_rate(regressed,regressed));
            PsiWish_alphasum = 0;
            for n = 1:sum(regressed)
                PsiWish_alphasum = PsiWish_alphasum+0.5*psi(hmm.state(k).Omega.Gam_shape/2+0.5-n/2);
            end
            C = hmm.state(k).Omega.Gam_shape * hmm.state(k).Omega.Gam_irate;
        end
    end
    
    if do_HMM_pca
        dist = - 0.5 * sum((XX / C) .* XX,2);

        % This is the expected loglik
        %M = W' * W + v * eye(p); % posterior dist of the precision matrix
        %iM = inv(M);
        %Zhat = XX * W * iM;
        %ZZhat = repmat(v * iM,[1 1 Tres]);
        %dist = -0.5 * iv * sum((XX.^2),2);
        %WW = W' * W;
        %S_W = zeros(p);
        %for n = 1:ndim
        %    S_W = S_W + permute(hmm.state(k).W.S_W(n,:,:),[2 3 1]);
        %end
        %for t = 1:Tres
        %    ZZhat(:,:,t) = ZZhat(:,:,t) + Zhat(t,:)' * Zhat(t,:);
        %    dist(t) = dist(t) - 0.5 * iv * trace(WW * ZZhat(:,:,t));
        %    dist(t) = dist(t) - 0.5 * iv * trace(S_W * ZZhat(:,:,t));
        %end
        %ZZhat = permute(ZZhat,[3 1 2]);
        %dist = dist + iv * sum( (Zhat * W') .* XX, 2) ...
        %    - 0.5 * sum(ZZhat(:,eye(p)==1),2);
        
    else
        meand = zeros(size(XX,1),sum(regressed));
        if train.uniqueAR
            for n = 1:ndim
                ind = n:ndim:size(XX,2);
                meand(:,n) = XX(:,ind) * hmm.state(k).W.Mu_W;
            end
        elseif ~isempty(hmm.state(k).W.Mu_W(:))
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
            dist = dist - 0.5 * (d(:,n).*Cd(n,:)');
        end
    end
    
    NormWishtrace = zeros(Tres,1);
    if ~do_HMM_pca && ~isempty(hmm.state(k).W.Mu_W(:))
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
                    if all(S(:)==1) 
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
                    elseif any(S(:)~=1) && any(var(residuals(1:end-1,~regressed))~=0)
                        % time varying regressors - this inference will be
                        % exceedingly slow, can be optimised if necessary
                        validentries = logical(S(:));
                        B_S = hmm.state(k).W.S_W(validentries,validentries);
                        for iT=1:Tres
                            NormWishtrace(iT) = trace(kron(C(regressed,regressed),...
                                residuals(1,~regressed)'*residuals(1,~regressed))*B_S);
                        end
                    else
                        % implies a static regressor value over full course of each trial - no time varying component:
                        if ~isempty(WishTrace)
                            Xval = residuals(1,~regressed)*[2.^(1:sum(~regressed))]';
                            NormWishtrace = repmat(WishTrace(cache.codevals==Xval),Tres,1);
                        else
                            validentries = logical(S(:));
                            B_S = hmm.state(k).W.S_W(validentries,validentries);
                            normtrace = trace(kron(C(regressed,regressed),residuals(1,~regressed)'*residuals(1,~regressed))*B_S);
                            NormWishtrace = repmat(normtrace,Tres,1); 
                        end
                    end
                end
        end
    end
    
    L(hmm.train.maxorder+1:T,k)= - ltpi - ldetWishB + PsiWish_alphasum + dist - NormWishtrace; 
end
% correct for stability problems by adding constant:
if any(all(L<0,2))
    L(all(L<0,2),:) = L(all(L<0,2),:) - repmat(max(L(all(L<0,2),:),[],2),1,K);
end
L = exp(L);
end
