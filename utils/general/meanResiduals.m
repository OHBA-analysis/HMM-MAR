function [mean_residuals,demean_residuals] = meanResiduals(hmm,Gamma,orders,XX,residuals)
 
K = length(hmm.state); T = size(Gamma,1);
S = hmm.train.S==1;
regressed = sum(S,1)>0;
if isempty(orders), 
    order = 0; 
else
    order = orders(end);
end

residuals0 = residuals;
if order>0
    for k=1:K
        residuals(:,regressed) = residuals(:,regressed) - repmat(Gamma(:,k),1,sum(regressed)) .* (XX * hmm.state(k).W.Mu_W(:,regressed)); 
    end;
end
mean_residuals = zeros(size(residuals));

for k=1:K,
    mean_residuals(:,regressed) = mean_residuals(:,regressed) + repmat(hmm.state(k).Mean.Mu(regressed)',T,1) .* repmat(Gamma(:,k),1,sum(regressed)) ;
end

demean_residuals = residuals0 - mean_residuals;

end