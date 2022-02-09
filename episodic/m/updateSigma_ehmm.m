function ehmm = updateSigma_ehmm(ehmm,rangeK)
if nargin < 2 || isempty(rangeK), rangeK = 1:ehmm.K; end % K+1
ehmm = updateSigma(ehmm,rangeK);
end
