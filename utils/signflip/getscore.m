function score = getscore(M)
% get the score from the (flipped) matrices of autocorrelation contained in M
% M is a matrix (ndim x ndim x lags x subjects) with autocovariances matrix for all subjects
N = size(M,4); ndim = size(M,1); L = size(M,3);
M = reshape(M,[(ndim^2)*L N]);
C = corr(M);
score = mean(C(triu(true(N),1)));
end