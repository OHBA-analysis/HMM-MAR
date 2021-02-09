function ness = updateSigma_ness(ness,rangeK)
if nargin < 2 || isempty(rangeK), rangeK = 1:ness.K+1; end
ness = updateSigma(ness,rangeK);
end
