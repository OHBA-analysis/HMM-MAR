function [dataout,T] = simTIDEdata(T,Sigma,embeddedlags,nCh,burninpoints)
% this function generates data from a time delay embedded model. It
% requires the following inputs:
%
% inputs:
%      T    The timing of the chain to be simulated (in the standard HMM
%           format)
%      Sigma The time-lagged autocovariance matrix from which data is to be
%           simulated
%      embeddedlags The specified time lags at each dimension of the matrix
%           Sigma
%      nCh   The number of channels being simulated
%      burninpoints (optional) - a set of points from which to start the
%           simulation. This can be included from real data, in which case
%           the first N points of dataout will be exactly equal to these;
%           otherwise we generate the first N points froma  standard normal
%           distribution, and discard all data from a necessary burn-in
%           period.
%
% outputs:
%   dataout The simulated data
%   T       The timing information.
%
% 
%  Author: Cam Higgins, 2021

ndim = size(Sigma,1);
if size(Sigma,2)~=ndim && size(Sigma,3)~=1
    error('Covarance matrix must be square')
end
if length(embeddedlags)~=ndim
    error('embeddedlags should specify the timelag of EVERY channel in the matrix (and must therefore be the same length as the matrix)');
end
M = ndim/nCh;
if M~=floor(M)
    error('Requires same embedding length on every channel')
end

% ensure embeddedlags are uniform and equal to 1:
if any(diff(sort(unique(embeddedlags)))~=1)
    error('Embeddedlags must be unique and increment by one timestep only');
end

% check if matrix needs to be rearranged:
if length(unique(embeddedlags(1:(ndim/nCh))))>1 || embeddedlags(end)~=min(embeddedlags)
    inds = [];newembeddedlags = [];
    for i=max(embeddedlags):-1:min(embeddedlags)
        temp = find(embeddedlags==i);
        inds = [inds,temp];
        newembeddedlags = [newembeddedlags,embeddedlags(temp)];
    end
    Sigma = Sigma(inds,inds);
    embeddedlags = newembeddedlags;
end

% compute inverse:
Precisionmat = inv(Sigma);
%Lambda_pre_t = Precisionmat(1:nCh*(M-1),1:nCh*(M-1));
Lambda_cross = -Precisionmat(1:nCh*(M-1),(1+nCh*(M-1)):end);
Lambda_self = Precisionmat((1+nCh*(M-1)):end,(1+nCh*(M-1)):end);

S = inv(Lambda_self);
A = chol(S); % now A*Z is ~ MVN(0,S)
m_to_mult = Lambda_cross*S;

n_burnin = 400;
T(T<n_burnin) = []; % don't allow short segments without sufficient burn-in time

dataout = zeros(sum(T),nCh);
% now simulate the chains specified in T:
for i=1:length(T)
    Z = randn(T(i),nCh);
    if nargin==5
        Z(1:M-1,:) = burninpoints;
    end
    %Z = mvnrnd(zeros(1,nCh),S,T(i));
    for t=M:T(i)
        %Z(t,:) = Z(t,:)*A + reshape(Z(t-M+1:t-1,:),[1,(M-1)*nCh]) * m_to_mult;
        Z(t,:) = Z(t,:)*A + reshape(Z(t-M+1:t-1,:)',[1,(M-1)*nCh]) * m_to_mult;
    end
    dataout(1+[sum(T(1:(i-1))):(sum(T(1:i)))-1],:) = Z;
end

% remove burn-in points:
burnwindow = 30;
v = nan(length(dataout),1);
for i=1:length(T)
    for t=(1+sum(T(1:(i-1)))+burnwindow):sum(T(1:i))
        v(t) = sum(var(dataout((t-burnwindow+1):t,:),[],2));
    end
end
% remove burn-in points - find where v reaches 50th percentile
to_remove = false(length(dataout),1);
for i=1:length(T)
    segment = (1+sum(T(1:(i-1)))):sum(T(1:i));
    burnin_complete = find(v(segment)<prctile(v,50),1);
    to_remove(segment(1):segment(1)+burnin_complete-1) = true;
    T(i) = T(i) - burnin_complete;
end
dataout(to_remove,:) = [];
end