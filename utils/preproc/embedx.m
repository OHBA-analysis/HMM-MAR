function [Xe,valid] = embedx (X,lags)
%  
% Embeds a [samples x channels] array X using a vector of time lags 
% resulting in a [samples x (channels x lags)] array Xe. The embedded array 
% Xe is structured as:
%
% Xe = [channel 1,   lag 1
%       channel 1,   lag 2
%       channel 1,   lag ...
%       channel 1,   lag N
%       channel 2,   lag 1
%       channel 2,   lag 2
%       channel 2,   lag ...
%       channel 2,   lag N
%       ....................
%       ....................
%       ....................
%       ....................
%       channel M,   lag 1
%       channel M,   lag 2
%       channel M,   lag ...
%       channel M,   lag N
%
% Xe contains only the valid subsection of the orignal data, after edge
% effects have been removed. The indices of the valid samples from X are
% optionally output in the vector valid.
%
% Romesh Abeysuriya 2017
% Author: Adam Baker, OHBA, University of Oxford


Xe = zeros(size(X,1),size(X,2)*length(lags));

for l = 1:length(lags)
    Xe(:,l:length(lags):end) = circshift(X,lags(l));
end

valid = true(size(X,1),1);
valid(end+min(lags)+1:end) = 0;
valid(1:max(lags)) = 0;

Xe = Xe(valid,:);
