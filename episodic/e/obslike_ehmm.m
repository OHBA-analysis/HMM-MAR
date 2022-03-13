function L = obslike_ehmm(ehmm,Gamma,residuals,XX,k)
%
% Evaluate likelihood of data given observation model
% for chain k, for one continuous trial
% (It's pretty much the same than obslike but simplified)
%
% INPUT
% X          N by ndim data matrix
% ehmm        ehmm data structure
% residuals  in case we train on residuals, the value of those.
% XX        alternatively to X (which in this case can be specified as []),
%               XX can be provided as computed by setxx.m
% OUTPUT
% B          Likelihood of N data points
%
% Author: Diego Vidaurre, University of Oxford / Aarhus University (2022)

K = ehmm.K;

[T,ndim] = size(residuals);
setstateoptions;
ltpi = sum(regressed)/2 * log(2*pi);
L = zeros(T+ehmm.train.maxorder,2);

ldetWishB = 0;
PsiWish_alphasum = 0;
for n = 1:ndim % only diagonal? 
    if ~regressed(n), continue; end
    ldetWishB = ldetWishB+0.5*log(ehmm.Omega.Gam_rate(n));
    PsiWish_alphasum = PsiWish_alphasum+0.5*psi(ehmm.Omega.Gam_shape);
end
C = ehmm.Omega.Gam_shape ./ ehmm.Omega.Gam_rate;

for l = 1:2
        
    if l == 1, Gamma(:,k) = 1; 
    else, Gamma(:,k) = 0; 
    end

    [Xhat,XXstar] = computeStateResponses(XX,ehmm,Gamma,1:K+1);

    d = residuals(:,regressed) - Xhat(:,regressed);
    Cd = bsxfun(@times,C(regressed),d)';
    dist = zeros(T,1);
    for n = 1:sum(regressed)
        dist = dist - 0.5 * (d(:,n).*Cd(n,:)');
    end
     
    NormWishtrace = zeros(T,1);
%     if ndim == 1
%           NormWishtrace = NormWishtrace + 0.5 * C * ...
%                 sum( (XXstar * ehmm.state_shared(1).W.S_W) .* XXstar, 2);
%     else
%         for n = 1:ndim
%             if ~regressed(n), continue; end
%             Sind_all = repmat(Sind(:,n),K+1,1) == 1;
%             NormWishtrace = NormWishtrace + 0.5 * C(n) * ...
%                 sum( (XXstar(:,Sind_all) * ehmm.state_shared(n).S_W(Sind_all,Sind_all)) ...
%                 .* XXstar(:,Sind_all), 2);
%         end
%     end
    
    L(ehmm.train.maxorder+1:end,l) = - ltpi - ldetWishB + ...
        PsiWish_alphasum + dist - NormWishtrace;

end

Lneg = all(L<0,2);
if any(Lneg)
    L(Lneg,:) = L(Lneg,:) - repmat(max(L(Lneg,:),[],2),1,2);
end
%if any(sum(exp(L),2)==0), keyboard; end

L = exp(L);

end

