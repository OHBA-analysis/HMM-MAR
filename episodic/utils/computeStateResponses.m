function [Xhat,XXstar] = computeStateResponses(XX,ehmm,Gamma,states)

regressed = sum(ehmm.train.Sind==1,1)>0;
K = size(Gamma,2); np = size(XX,2); ndim = size(ehmm.train.S,1);

if nargin < 4 || isempty(states), states = 1:K+1; end

Gamma = [Gamma prod(1-Gamma,2) ];
Gamma = rdiv(Gamma,sum(Gamma,2));  

XXstar = zeros(size(XX,1),np * length(states));
ind = false((K+1)*np,1); 

for ik = 1:length(states)
    k = states(ik);
    XXstar(:,(1:np) + (ik-1)*np) = bsxfun(@times, XX, Gamma(:,k));
    ind( (1:np) + (k-1)*np) = true; 
end

Xhat = zeros(size(XX,1),ndim);
for n = find(regressed)
    Xhat(:,n) = XXstar * ehmm.state_shared(n).Mu_W(ind);
end

end