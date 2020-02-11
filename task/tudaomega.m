function [omega,omega_t] = tudaomega(tuda,Gamma_mean)
% Given a precomputed TUDA model, it returns the precision matrix of the
% noise (the inverse of the covariance matrix of the residuals)
%
% INPUT
% tuda: Estimated TUDA model, using tudatrain
% Gamma_mean: (Optional) if time-resolved coefficients are desired,
%    inputting the mean state timecourse here will output the average
%    time-resolved TUDA model coefficients.
%
% OUTPUT
% omega: (no. channels by no. channels by no. states) array of
%           precision matrices
% beta_t: (no. channels by no. channles by timepoints) array of
%           time-resolved precision matrices.
%
% Author: Cam Higgins, OHBA, University of Oxford (2019)

q = size(tuda.train.S,1) - find(tuda.train.S(1,:)>0,1) + 1;
p = find(tuda.train.S(1,:)>0,1) - 1; 
if ~(isempty(q) && isempty(p))
    omega = zeros(q,q,tuda.K);
    for k = 1:tuda.K
        if strcmp(tuda.train.covtype,'uniquediag')
            omega(:,:,k) = diag(tuda.Omega.Gam_shape ./ tuda.Omega.Gam_rate((p+1):(q+p)));
        elseif strcmp(tuda.train.covtype,'diag')
            omega(:,:,k) = diag(tuda.state(k).Omega.Gam_shape ./ tuda.state(k).Omega.Gam_rate((p+1):(q+p)));
        elseif strcmp(tuda.train.covtype,'uniquefull')
            omega(:,:,k) = tuda.Omega.Gam_shape * tuda.Omega.Gam_irate((p+1):(q+p),(p+1):(q+p));
        elseif strcmp(tuda.train.covtype,'full')
            omega(:,:,k) = tuda.state(k).Omega.Gam_shape * tuda.state(k).Omega.Gam_irate((p+1):(q+p),(p+1):(q+p));
        end
    end
else % LDA / LGS style setup:
    p = size(tuda.train.S',1) - find(tuda.train.S(:,1)>0,1) + 1;
    q = find(tuda.train.S(:,1)>0,1) - 1; 
    omega = zeros(q,q,tuda.K);
    for k = 1:tuda.K
        if strcmp(tuda.train.covtype,'uniquediag')
            omega(:,:,k) = diag(tuda.Omega.Gam_shape ./ tuda.Omega.Gam_rate(1:q));
        elseif strcmp(tuda.train.covtype,'diag')
            omega(:,:,k) = diag(tuda.state(k).Omega.Gam_shape ./ tuda.state(k).Omega.Gam_rate(1:q));
        elseif strcmp(tuda.train.covtype,'uniquefull')
            omega(:,:,k) = tuda.Omega.Gam_shape * tuda.Omega.Gam_irate(1:q,1:q);
        elseif strcmp(tuda.train.covtype,'full')
            omega(:,:,k) = tuda.state(k).Omega.Gam_shape * tuda.state(k).Omega.Gam_irate(1:q,1:q);
        end
    end
end

if nargin > 1
    T = size(Gamma_mean,1);
    omega_t = zeros(q,q,T);
    for t = 1:T
        omega_t(:,:,t) = sum(omega.*repmat(permute(Gamma_mean(t,:),[3,1,2]),q,q,1),3);
    end
end

end