function L = obslike_addHMM(hmm,residuals,XX,rangeK)
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

K = hmm.K;
if nargin < 4 || isempty(rangeK), rangeK = 1:K; end

[T,ndim] = size(residuals);
S = hmm.train.S==1;
regressed = sum(S,1)>0;
ltpi = sum(regressed)/2 * log(2*pi);
L = zeros(T,length(rangeK));

ldetWishB = 0;
PsiWish_alphasum = 0;
for n = 1:ndim
    if ~regressed(n), continue; end
    ldetWishB = ldetWishB+0.5*log(hmm.Omega.Gam_rate(n));
    PsiWish_alphasum = PsiWish_alphasum+0.5*psi(hmm.Omega.Gam_shape);
end
C = hmm.Omega.Gam_shape ./ hmm.Omega.Gam_rate;

setstateoptions;

for ik = 1:length(rangeK)
    
    k = rangeK(ik);

    meand = XX * hmm.state(k).W.Mu_W(:,regressed);
    d = residuals(:,regressed) - meand;
    Cd = bsxfun(@times,C(regressed),d)';
    dist = zeros(T,1);
    for n = 1:sum(regressed)
        dist = dist - 0.5 * (d(:,n).*Cd(n,:)');
    end
    
    % Covariance of the distance
    NormWishtrace = zeros(T,1);  
    for n = 1:ndim
        if ~regressed(n), continue; end
        if ndim==1
            NormWishtrace = NormWishtrace + 0.5 * C(n) * ...
                sum( (XX(:,Sind(:,n)) * hmm.state(k).W.S_W) ...
                .* XX(:,Sind(:,n)), 2);
        else
            NormWishtrace = NormWishtrace + 0.5 * C(n) * ...
                sum( (XX(:,Sind(:,n)) * ...
                permute(hmm.state(k).W.S_W(n,Sind(:,n),Sind(:,n)),[2 3 1])) ...
                .* XX(:,Sind(:,n)), 2);
        end
    end
    
    L(hmm.train.maxorder+1:end,ik)= - ltpi - ldetWishB + PsiWish_alphasum + dist - NormWishtrace;
end
% correct for stability problems by adding constant:
if any(all(L<0,2))
    L(all(L<0,2),:) = L(all(L<0,2),:) - repmat(max(L(all(L<0,2),:),[],2),1,length(rangeK));
end
L = exp(L);
end

