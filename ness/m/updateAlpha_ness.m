function ness = updateAlpha_ness(ness,rangeK)
if nargin < 2 || isempty(rangeK), rangeK = 1:ness.K; end %+1
ness = updateAlpha(ness,rangeK);
end