function C_sq = weighted_covariance(x,w)
%
% Computes the covariance matrix for a set of observations x given 
% weights in w. See https://www.wikipedia.com/en/Weighted_arithmetic_mean#Weighted_sample_covariance
%
% X should be [N x P] and w should be [N x 1]
%
% C_sq, the covariance matrix, will be [P x P]
%
% Author: Cam Higgins, OHBA
%
if size(x,1)~=size(w,1)
    error('Weights and data must be same length');
end
if size(w,2)~=1
    error('Weights should be a vector of same length as data');
end

mu_w = sum(x.*repmat(w,[1,size(x,2)]))./sum(w);
xminusmu = x-repmat(mu_w,size(x,1),1);
C_sq = (xminusmu.*repmat(w,[1,size(xminusmu,2)]))'*xminusmu;
C_sq = C_sq ./ sum(w);

end