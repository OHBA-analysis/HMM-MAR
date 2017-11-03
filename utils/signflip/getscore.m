function [score,S] = getscore(M,n,S)
% get the score from the (flipped) matrices of autocorrelation contained in M
% M is a matrix (ndim x ndim x lags x subjects) with autocovariances matrix for all subjects
% S is a matrix (ndim x ndim x lags x subjects), such as S(:,:,:,n) contains
% the sum of M for all subjects excluding n 
N = size(M,4);
if nargin==1 
    sumM = sum(M,4);
    S = repmat(sumM,[1 1 1 N]) - M; % difference between the sum and each subject
else
    sumM = S(:,:,:,n) + M(:,:,:,n);
end
score = sum(abs(sumM(:)));
end