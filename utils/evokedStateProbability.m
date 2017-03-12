function evokedGamma = evokedStateProbability(stim,T,Gamma,window)
% Gets the evoked state relative to a stimulus
%
% INPUT
% stim          A (time x 1 = sum(T) x 1) boolean indicating when the stimulus occurs 
% T             Length of series
% Gamma         State time course
% Window        Window length, in time points, to be plotted around the stimulus
%
% OUTPUT
% evokedGamma   Evoked state time course, (window length by no. of states) 
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2017)

if mod(window,2)==0, 
    window = window + 1; 
end
if size(stim,1)==1, stim = stim'; end
if sum(T)~=length(stim) || size(stim,2)>1
    error('Argument stim was wrong dimensions - it needs to has the same elements as sum(T)')
end
order =  (sum(T) - size(Gamma,1)) / length(T);
K = size(Gamma,2);
halfwin = (window-1)/2; 
evokedGamma = zeros(window,K);
count = 0; 

for j = 1:length(T) % iterate through subjects/trials/segments
    ind1 = sum(T(1:j-1)) + (order+1:T(j));
    ind2 = sum(T(1:j-1)) - (j-1)*order + (1:T(j)-order);
    stim_j = stim(ind1);
    Gamma_j = Gamma(ind2,:);
    events = find(stim_j)';
    for i = events
        if (i-halfwin)>0 && (i+halfwin)<=size(Gamma_j,1)
            evokedGamma =  evokedGamma + Gamma_j(i - halfwin : i + halfwin,:);
            count = count + 1; 
        end
    end
end

evokedGamma = evokedGamma / count;

end