function meand = computeStateResponses(XX,hmm,Gamma,rangeK,baseline)
% rangeK - which chains to sum across
% baseline - do we include the baseline for the selected chains?

if nargin < 4, rangeK = 1:size(Gamma,2); end
if nargin < 5, baseline = true; end

S = hmm.train.S==1; 
ndim = length(S);
meand = zeros(size(XX,1),ndim);
regressed = sum(S,1)>0;

if ~isempty(rangeK)
    noGamma = zeros(size(Gamma,1),1);
    for k = rangeK % reaching until the complement of the sum of the other chains
        if hmm.train.uniqueAR
            for n = 1:ndim
                ind = n:ndim:size(XX,2);
                meand(:,n) = meand(:,n) + bsxfun(@times, XX(:,ind) * hmm.state(k).W.Mu_W, Gamma(:,k));
            end
        elseif ~isempty(hmm.state(k).W.Mu_W(:))
            meand = meand + bsxfun(@times, XX * hmm.state(k).W.Mu_W(:,regressed), Gamma(:,k));
        end
        if baseline
            noGamma = noGamma + (1-Gamma(:,k));
        end
    end
else
    noGamma = size(Gamma,2) - sum(Gamma,2);
end

if baseline
    if hmm.train.uniqueAR
        for n = 1:ndim
            ind = n:ndim:size(XX,2);
            meand(:,n) = meand(:,n) + bsxfun(@times, XX(:,ind) * hmm.state(end).W.Mu_W, noGamma);
        end
    else
        meand = meand + bsxfun(@times, XX * hmm.state(end).W.Mu_W(:,regressed), noGamma);
    end
end

end


