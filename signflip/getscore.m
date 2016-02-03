function score = getscore(M)
% get the score from the (flipped) matrices of autocorrelation contained in M
sumM = abs(sum(M,4));
score = sum(sumM(:));
end