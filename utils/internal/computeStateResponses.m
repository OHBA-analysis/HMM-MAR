function meand = computeStateResponses(XX,ndim,hmm,Gamma,rangeK,baseline)
% rangeK - which chains to sum across
% baseline - do we include the baseline for the selected chains?

meand = zeros(size(XX,1),ndim);
S = hmm.train.S==1; 
regressed = sum(S,1)>0;

for k = rangeK % reaching until the complement of the sum of the other chains
    if hmm.train.uniqueAR
        for n = 1:ndim
            ind = n:ndim:size(XX,2);
            meand(:,n) = meand(:,n) + bsxfun(@times, XX(:,ind) * hmm.state(k).W.Mu_W, Gamma(:,k));
        end
    elseif ~isempty(hmm.state(k).W.Mu_W(:))
        meand = meand + bsxfun(@times, XX * hmm.state(k).W.Mu_W(:,regressed), Gamma(:,k));
    end
end

if baseline
    Gamma = length(rangeK) * ones(size(Gamma,1),1) - sum(Gamma(:,rangeK),2);
    meand  = meand + bsxfun(@times, XX * hmm.state(end).W.Mu_W(:,regressed), Gamma);
end

end


