function hmm = updateOmega_addHMM(hmm,Gamma,residuals,T,XX,Tfactor)

K = hmm.K; ndim = hmm.train.ndim;
if nargin < 6, Tfactor = 1; end
S = hmm.train.S==1; regressed = sum(S,1)>0;
Tres = sum(T) - length(T)*hmm.train.maxorder;
np = size(XX,2);

setstateoptions;

meand = computeStateResponses(XX,hmm,Gamma);
e = (residuals - meand).^2;

Gamma = [Gamma (size(Gamma,2)-sum(Gamma,2)) ];
swx2 = zeros(Tres,ndim);
if ~isempty(hmm.state(1).W.Mu_W)
    X = zeros(size(XX,1),np * (K+1));
    for k = 1:K+1
        X(:,(1:np) + (k-1)*np) = bsxfun(@times, XX, Gamma(:,k));
    end
    for n = 1:ndim
        if ~regressed(n), continue; end
        Sind_all = [];
        for k = 1:K+1
            Sind_all = [Sind_all; Sind(:,n)];
        end
        Sind_all = Sind_all == 1; 
        tmp = X(:,Sind_all) * hmm.state_shared(n).Mu_W;
        swx2(:,n) = sum(tmp .* X(:,Sind_all),2);
    end
end
     
hmm.Omega.Gam_rate(regressed) = hmm.prior.Omega.Gam_rate(regressed) + ...
    0.5 * Tfactor * sum(e + swx2(:,regressed)); 

hmm.Omega.Gam_shape = hmm.prior.Omega.Gam_shape + 0.5 * Tfactor * Tres;


end
