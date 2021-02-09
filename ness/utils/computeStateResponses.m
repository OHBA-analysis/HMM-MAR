function [meand,X] = computeStateResponses(XX,ness,Gamma)

K = size(Gamma,2); np = size(XX,2); ndim = length(ness.state_shared); 
Gamma = [Gamma (K-sum(Gamma,2)) ];
X = zeros(size(XX,1),np * (K+1));
for k = 1:K+1
   X(:,(1:np) + (k-1)*np) = bsxfun(@times, XX, Gamma(:,k)); 
end

meand = zeros(size(XX,1),ndim);
for n = 1:ndim
    meand(:,n) = X * ness.state_shared(n).Mu_W;
end

end


