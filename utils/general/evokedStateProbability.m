function [evokedGamma,t] = evokedStateProbability(stimulus,T,Gamma,window,Fs,downsample)
% Gets the evoked state relative to a stimulus
%
% INPUT
% stim          A (time x 1 = sum(T) x 1) boolean indicating when the stimulus occurs 
% T             Length of series
% Gamma         State time course
% Window        Window length, in seconds, to be plotted around the stimulus
% Fs            Sampling frequency  of the data / Gamma / stimulus
% downsample    if the HMM was trained with option .downsample, the same
%               should go here (by default =Fs) 
%
% OUTPUT
% evokedGamma   Evoked state time course, (window length by no. of states) 
% t             time indexes
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2017)

window = round(window * Fs); 
if mod(window,2)==0
    window = window + 1; 
end
if size(stimulus,1)==1, stimulus = stimulus'; end
if sum(T)~=length(stimulus) || size(stimulus,2)>1
    error('Argument stim was wrong dimensions - it needs to has the same elements as sum(T)')
end

if nargin==6 && Fs~=downsample
    [stimulus,T] = downsampledata(stimulus,T,downsample,Fs);
end

order =  (sum(T) - size(Gamma,1)) / length(T);
K = size(Gamma,2);
halfwin = (window-1)/2; 
evokedGamma = zeros(window,K);
count = 0; 
t = (-halfwin : halfwin)';
t = t / Fs;

for j = 1:length(T) % iterate through subjects/trials/segments
    ind1 = sum(T(1:j-1)) + (order+1:T(j));
    ind2 = sum(T(1:j-1)) - (j-1)*order + (1:T(j)-order);
    stim_j = stimulus(ind1);
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