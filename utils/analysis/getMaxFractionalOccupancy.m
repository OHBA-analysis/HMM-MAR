function maxFO = getMaxFractionalOccupancy(Gamma,T,options)
% Finds the maximum fractional occupancy for each subject/trial.
% This is a useful statistic to diagnose whether the HMM solution is
% "mixing well" or states are assigned to describe entire subjects (in
% which case the HMM is doing a poor job in finding the data dynamics)
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2017)

if nargin<3, options = struct(); end
FO = getFractionalOccupancy(Gamma,T,options,2);
maxFO = max(FO,[],2);

end
