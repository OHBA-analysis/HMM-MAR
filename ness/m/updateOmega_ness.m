function hmm = updateOmega_ness(hmm,Gamma,residuals,T,XX,Tfactor)
% not implemented for anything else than uniquediag

K = hmm.K; ndim = hmm.train.ndim;
if nargin < 6, Tfactor = 1; end
S = hmm.train.S==1; regressed = sum(S,1)>0;
Tres = sum(T) - length(T)*hmm.train.maxorder;
np = size(XX,2);

setstateoptions;

[meand,X] = computeStateResponses(XX,hmm,Gamma);
e = (residuals - meand).^2;
swx2 = zeros(Tres,ndim);
for n = 1:ndim
    if ~regressed(n), continue; end
    Sind_all = [];
    for k = 1:K+1
        Sind_all = [Sind_all; Sind(:,n)];
    end
    Sind_all = Sind_all == 1;
    tmp = X(:,Sind(:,n)) * hmm.state_shared(n).S_W(Sind(:,n),Sind(:,n));
    swx2(:,n) = sum(tmp .* X(:,Sind(:,n)),2);
end

hmm.Omega.Gam_rate(regressed) = hmm.prior.Omega.Gam_rate(regressed) + ...
    0.5 * Tfactor * sum(e + swx2(:,regressed)); 
hmm.Omega.Gam_shape = hmm.prior.Omega.Gam_shape + 0.5 * Tfactor * Tres;

end
