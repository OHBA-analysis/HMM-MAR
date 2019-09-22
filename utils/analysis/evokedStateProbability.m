function [evokedGamma,t] = evokedStateProbability(stimulus,T,Gamma,window,options)
% Gets the evoked state relative to a stimulus 
%
% INPUT
% stim          A (time x 1 = sum(T) x 1) boolean indicating when the stimulus occurs
% T             Length of series
% Gamma         State time course
% window        Window length, in seconds, to be plotted around the stimulus
% options       parameters used to train the HMM (should include at least .Fs)

% OUTPUT
% evokedGamma   Evoked state time course, (window length by no. of states) 
% t             time indexes
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2017)

if nargin<5, options = struct(); options.Fs = 1; options.downsample = 1; end
if ~isfield(options,'Fs'), options.Fs = 1; end
if ~isfield(options,'downsample'), options.downsample = options.Fs; end

downsampling = (options.Fs~=options.downsample) && (options.downsample>0);
N = length(T);

window = round(window * options.Fs); 
if mod(window,2)==0
    window = window + 1; 
end
if size(stimulus,1)==1, stimulus = stimulus'; end
if size(stimulus,2)>1
    error('Argument stimulus was wrong dimensions - it needs to has one column')
end
if (sum(T)~=length(stimulus)) && (size(Gamma,1)~=length(stimulus))
    error('Argument stimulus was wrong dimensions - it needs to has the same elements as sum(T)')
end
if length(unique(stimulus))>2
    error('stimulus must be made of 0s and 1s')
end
 
Gamma = padGamma(Gamma,T,options);

if downsampling
    [stimulus,T] = downsampledata(stimulus,T,options.downsample,options.Fs);
end
    
K = size(Gamma,2);
halfwin = (window-1)/2; 
evokedGamma = zeros(window,K);
count = zeros(window,1); 
t = (-halfwin : halfwin)';
t = t / options.Fs;

for j = 1:N % iterate through subjects/trials/segments
    jj = sum(T(1:j-1)) + (1:T(j));
    stim_j = stimulus(jj);
    Gamma_j = Gamma(jj,:);
    events = find(stim_j)';
    for i = events
        ind = true(window,1);
        if i-halfwin-1<0, ind(1:halfwin-i+1) = false; end
        if i+halfwin>T(j), e = i+halfwin-T(j); ind(end-e+1:end) = false; end
        t1 = max(1,i-halfwin); t2 = min(T(j),i+halfwin);
        evokedGamma(ind,:) = evokedGamma(ind,:) + Gamma_j(t1:t2,:);
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