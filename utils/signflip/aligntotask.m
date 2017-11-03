function [X,flips] = aligntotask(X,T,events,window)
% Assuming that the sign ambiguity across channels has been resolved using
% findflip/flipdata.m, we might need to go one step further, under 
% the condition that we have a task or a stimulus to lock to.
% If our interest is for example to get the stimulus-locked average of
% the evoked response, then we also need the sessions or subjects to be
% consistently aligned with respect to each other, as otherwise the evoked
% response will cancel;
% aligntotask.m does this using the information of the task

% INPUTS
% X             time series, or alternatively an (unflipped) array of
%                   autocorrelation matrices (ndim x ndim x no.lags x no. trials),
%                   as computed for example by getCovMats()
% T             length of series
% events         a vector with the positions of each time the event happens in the data a (time by 1) 
%               logical vector indicating when the evoked response is happening. 
% window        a (1 by 2) vector - 
%               first component: tells when the evoked response starts,  
%                   in no. of time points, with respect to the event   
%                   (a negative number indicates that it starts before the event); 
%               second component: how long, in no. of time points, 
%                   the evoked response lives after the event
%
% OUTPUT
% X             time series, after being aligned
% flips         (length(T) X 1) binary vector saying which segments must be flipped 
%
% Author: Diego Vidaurre, University of Oxford.

p = size(X,2);  
N = length(T);

if size(events,2)==1; events = events'; end

Y = zeros(window(2)-window(1)+1,p,N);
with_event = false(1,N);
flips = zeros(N,1);

for i=1:N
    t = (1:T(i)) + sum(T(1:i-1));
    these = events(events>t(1) & events<t(end)); 
    if isempty(these), continue; end % no event in this one
    with_event(i) = true;
    c = 0;
    for event=these
        if (event+window(2))>sum(T(1:i)), continue; end % event too late
        if (event-window(1))<sum(T(1:i-1))+1, continue; end % event too early
        Y(:,:,i) = Y(:,:,i) + X(event+window(1):event+window(2),:);
        c = c + 1;
    end
    Y(:,:,i) = Y(:,:,i) / c; 
    if sum(with_event(1:i-1))==0, continue; end % no need for sign flipping
    Z = sum(Y(:,:,with_event(1:i-1)),3);
    Yi = Y(:,:,i);
    if sum(abs(Z(:) + Yi(:))) < sum(abs(Z(:) - Yi(:))) % flip segment
       Y(:,:,i) = - Y(:,:,i);
       X(t,:) = - X(t,:);
       flips(i) = 1;
    end
end

end


