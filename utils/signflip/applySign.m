function CovMats = applySign(CovMats,SignMats)
% apply matrices of sign flipping to the autocorrelation matrices
N = size(SignMats,3); nlags = size(CovMats,3);
for j = 1:N
    CovMats(:,:,:,j) = CovMats(:,:,:,j) .* repmat(SignMats(:,:,j),[1 1 nlags]);
end
end