function ness = updateAlpha_ness(ness,rangeK)
if nargin < 2 || isempty(rangeK), rangeK = 1:ness.K+1; end
ness = updateAlpha(ness,rangeK);
end