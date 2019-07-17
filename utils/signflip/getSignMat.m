function SignMats = getSignMat(Flips)
% Construct matrices of sign flipping for the autocorrelation matrices
[N,ndim] = size(Flips);
SignMats = zeros(ndim,ndim,N);
for j = 1:N 
    flips = ones(1,ndim);
    flips(Flips(j,:)==1) = -1;
    SignMats(:,:,j) = flips' * flips;
end
end