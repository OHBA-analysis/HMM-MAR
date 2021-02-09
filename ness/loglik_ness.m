function loglik = loglik_ness(XX,residuals,T,ness)
%
% Evaluate log likelihood of the model
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

K = ness.K; setstateoptions;
N = length(T);
ndim = size(residuals,2);
S = ness.train.S==1;
regressed = sum(S,1)>0;
ltpi = sum(regressed)/2 * log(2*pi);

ldetWishB = 0;
PsiWish_alphasum = 0;
for n = 1:ndim % only diagonal?
    if ~regressed(n), continue; end
    ldetWishB = ldetWishB+0.5*log(ness.Omega.Gam_rate(n));
    PsiWish_alphasum = PsiWish_alphasum+0.5*psi(ness.Omega.Gam_shape);
end
C = ness.Omega.Gam_shape ./ ness.Omega.Gam_rate;

combinations = zeros(2^K,K);
for k = 1:K
    combinations(:,k) = repmat([2*ones(2^(k-1),1); ones(2^(k-1),1)],2^(K-k),1);
end % 1 is active, 2 is baseline

Pbig = ones(2^K); Pibig = zeros(1,2^K);

for c1 = 1:2^K
    p1 = combinations(c1,:);
    for c2 = 1:2^K
        p2 = combinations(c2,:);
        for k = 1:K
            Pbig(c1,c2) = Pbig(c1,c2) * ness.state(k).P(p1(k),p2(k));
        end
    end
end
Pibig(end) = 1;

combinations = zeros(2^K,K);
for k = 1:K
    combinations(:,k) = repmat([2*ones(2^(k-1),1); ones(2^(k-1),1)],2^(K-k),1);
end % 1 is active, 0 is not active

loglik = zeros(N,1);

for j = 1:N
    
    t0 = sum(T(1:j-1)) - (j-1)*order + 1;
    t1 = sum(T(1:j)) - j*order;
    
    XXj = XX(t0:t1,:);
    residualsj = residuals(t0:t1,:);
    
    L = zeros(T(j)-order,2^K);
    
    for c = 1:2^K
        Gamma_c = zeros(size(XXj,1),K);
        for k = 1:K
            Gamma_c(:,k) = combinations(c,k);
        end
        
        % compute the mean response
        [meand,X] = computeStateResponses(XX,ness,Gamma_c);
        d = residualsj(:,regressed) - meand;
        Cd = bsxfun(@times, C(regressed), d)';
        dist = zeros(T(j)-order,1);
        for n = 1:sum(regressed)
            dist = dist - 0.5 * (d(:,n).*Cd(n,:)');
        end
        % Covariance of the distance
        NormWishtrace = zeros(T(j)-order,1);
        for n = 1:ndim
            if ~regressed(n), continue; end
            Sind_all = [];
            for k = 1:K+1
                Sind_all = [Sind_all; Sind(:,n)];
            end
            Sind_all = Sind_all == 1;
            if ndim==1
                NormWishtrace = NormWishtrace + 0.5 * C(n) * ...
                    sum( (X(:,Sind_all) * ness.state_shared(n).S_W(Sind_all,Sind_all)) ...
                    .* X(:,Sind_all), 2);
            else
                NormWishtrace = NormWishtrace + 0.5 * C(n) * ...
                    sum( (X(:,Sind_all) * ...
                    ness.state_shared(n).S_W(Sind_all,Sind_all)) ...
                    .* X(:,Sind_all), 2);
            end
        end
        
        L(:,c) = - ltpi - ldetWishB + PsiWish_alphasum + dist - NormWishtrace;
    end
    
    L = exp(L);
    
    scale = zeros(T(j)-order,1);
    alpha = zeros(T(j)-order,2^K);
    
    alpha(1,:) = Pibig.*L(1,:);
    scale(1) = sum(alpha(1,:));
    alpha(1,:) = alpha(1,:)/(scale(1)+realmin);
    for i = 2:T(j)-order
        alpha(i,:) = (alpha(i-1,:)*Pbig).*L(i,:);
        scale(i) = sum(alpha(i,:));		% P(X_i | X_1 ... X_{i-1})
        alpha(i,:) = alpha(i,:)/(scale(i)+realmin);
    end
    
    if isnan(sum(log(scale+realmin))), keyboard; end
    
    loglik(j) = sum(log(scale+realmin));
    
end


end

