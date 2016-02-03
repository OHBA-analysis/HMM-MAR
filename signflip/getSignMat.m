function SignMats = getSignMat(Flips)
% Construct matrices of sign flipping for the autocorrelation matrices
[N,ndim] = size(Flips);
SignMats = zeros(ndim,ndim,N);
for in=1:N 
    flips = ones(1,ndim);
    flips(Flips(in,:)==1) = -1;
    SignMats(:,:,in) = flips' * flips;
end
end