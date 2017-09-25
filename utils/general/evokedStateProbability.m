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
count = zeros(window,1); 
t = (-halfwin : halfwin)';
t = t / Fs;

for j = 1:length(T) % iterate through subjects/trials/segments
    jj1 = sum(T(1:j-1)) + (order+1:T(j));
    jj2 = sum(T(1:j-1)) - (j-1)*order + (1:T(j)-order);
    Tj = T(j)-order;
    stim_j = stimulus(jj1);
    Gamma_j = Gamma(jj2,:);
    events = find(stim_j)';
    for i = events
        ind = true(window,1);
        if i-halfwin-1<0, ind(1:halfwin-i+1) = false; end
        if i+halfwin>Tj, e = i+halfwin-Tj; ind(end-e+1:end) = false; end
        t1 = max(1,i-halfwin); t2 = min(Tj,i+halfwin);
        try
        evokedGamma(ind,:) = evokedGamma(ind,:) + Gamma_j(t1:t2,:);
        catch, keyboard; end
        count(ind) = count(ind) + 1; 
        %if (i-halfwin)>0 && (i+halfwin)<=size(Gamma_j,1)
        %    evokedGamma =  evokedGamma + Gamma_j(i - halfwin : i + halfwin,:);
        %    if any(isnan(evokedGamma(:))), keyboard; end
        %    count = count + 1; 
        %end
    end
end

jj = count>0;
evokedGamma(jj,:) = evokedGamma(jj,:) ./ repmat(count(jj),1,K);

end