function s = getSynchronicity(Gamma,T)
% Returns a time resolved measure of synchronicity, computed as the 
% maximum fractional occupancy across trials, i.e. for each time point   
% the across-trial average loading of the state that is more dominant 
% at this time point 
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2018)

if ~all(T(1)==T)
    error('Synchronisity can only be measured if trials have the same length'); 
end 
N = length(T); 
ttrial = size(Gamma,1)/N;
K = size(Gamma,2);
g = permute(mean(reshape(Gamma,[ttrial N K]),2),[1 3 2]);
s = max(g,[],2);

end