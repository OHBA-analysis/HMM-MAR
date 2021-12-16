function ness = updateSigma_ness(ness,rangeK)
if nargin < 2 || isempty(rangeK), rangeK = 1:ness.K; end % K+1
ness = updateSigma(ness,rangeK);
end
