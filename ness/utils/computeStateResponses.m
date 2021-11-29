function [meand,X] = computeStateResponses(XX,ness,Gamma,baseline)

if nargin < 4 || isempty(baseline), baseline = 1; end 

S = hmm.train.S==1; regressed = sum(S,1)>0;
K = size(Gamma,2); np = size(XX,2); ndim = length(ness.state_shared); 
if nargin < 5, regressed = true(1,ndim); end 

if baseline
    Gamma = [Gamma prod(1-Gamma,2) ];
    %Gamma = [Gamma (K-sum(Gamma,2))/K ];
end

X = zeros(size(XX,1),np * (K+baseline));
% if nargout > 1, Xs = zeros(size(XX,1),np * K); end

for k = 1:K+baseline
   X(:,(1:np) + (k-1)*np) = bsxfun(@times, XX, Gamma(:,k));
%    if k > K, break; end
%    if nargout > 1
%        Xs(:,(1:np) + (k-1)*np) = bsxfun(@times, XX, sqrt(Gamma(:,k)));
%    end
end

meand = zeros(size(XX,1),ndim);
for n = find(regressed)
    if baseline
        W = [ness.state_shared(n).Mu_W; ness.state(end).W.Mu_W(:,n)];
    else
        W = ness.state_shared(n).Mu_W;
    end
    meand(:,n) = X * W;
end
meand = meand(regressed);

end


