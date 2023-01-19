function vp_surr = surrogate_vp(vp,K)
% Create surrogate the state visits from an estimated Viterbi path vp.
% While keeping the structure of the visits (i.e when exactly there is a
% change of state) and (roughly) the state fractional occupancies,
% in the surrogate vp_surr, the actual state visits (i.e. which state
% activates) is randomly sampled.
%
% Author: Diego Vidaurre, Aarhus University (2023) 

T = length(vp); 
Gamma = vpath_to_stc(vp,K);
cprob = cumsum(mean(Gamma));
t = 1;
vp_surr = zeros(size(vp));
while t <= T
    tnext = find(vp(t:end)~=vp(t),1); 
    if isempty(tnext), tnext = T + 1;
    else, tnext = t + tnext - 1; 
    end
    k = find(rand(1) <= cprob,1);
    vp_surr(t:tnext-1) = k;
    t = tnext; 
end
end


