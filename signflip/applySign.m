function CovMats = applySign(CovMats,SignMats)
% apply matrices of sign flipping to the autocorrelation matrices
N = size(SignMats,3); nlags = size(CovMats,3);
for in=1:N
    CovMats(:,:,:,in) = CovMats(:,:,:,in) .* repmat(SignMats(:,:,in),[1 1 nlags]);
end
end