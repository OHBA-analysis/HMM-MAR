function Entr = GammaEntropy_ehmm(Gamma,Xi,T)
% Entropy of the state time courses
Entr = 0; K = size(Gamma,2); order = (sum(T)-size(Gamma,1))/length(T);

for k = 1:K
    Gk = [Gamma(:,k) (1-Gamma(:,k))];
    Xik = permute(Xi(:,k,:,:),[1 3 4 2]);
    Entr = Entr + GammaEntropy(Gk,Xik,T,order);
end

if isnan(Entr(:)) 
    error(['Error computing entropy of the state time courses  - ' ...
        'Out of precision?'])     
end
end
