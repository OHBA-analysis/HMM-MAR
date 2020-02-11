function [beta,beta_t] = tudabeta(tuda,Gamma_mean)
% Given a precomputed TUDA model, it returns the decoding coefficients
% (aka the "beta" parameters) 
%
% INPUT
% tuda: Estimated TUDA model, using tudatrain
% Gamma_mean: (Optional) if time-resolved coefficients are desired,
%    inputting the mean state timecourse here will output the average
%    time-resolved TUDA model coefficients.
%
% OUTPUT
% beta: (no. channels by no. predicted stimuli by no. states) array of
%           decoding coefficients
% beta_t: (no. channels by no. predicted stimuli by timepoints) array of
%           time-resolved decoding coefficients.
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2018)

q = size(tuda.train.S,1) - find(tuda.train.S(1,:)>0,1) + 1;
p = find(tuda.train.S(1,:)>0,1) - 1; 
if ~(isempty(p) && isempty(q)) %standard tuda setup
    beta = zeros(p,q,tuda.K);
    for k = 1:tuda.K
        beta(:,:,k) = tuda.state(k).W.Mu_W(1:p,p+1:end);
    end
else % LDA or LGS setup
    p = size(tuda.train.S',1) - find(tuda.train.S(:,1)>0,1) + 1;
    q = find(tuda.train.S(:,1)>0,1) - 1; 
    beta = zeros(p,q,tuda.K);
    for k = 1:tuda.K
        beta(:,:,k) = tuda.state(k).W.Mu_W(q+1:end,1:q);
    end
end
%beta = squeeze(beta); 

if nargin > 1
    T = size(Gamma_mean,1);
    beta_t = zeros(p,q,T);
    for t = 1:T
        beta_t(:,:,t) = sum(beta.*repmat(permute(Gamma_mean(t,:),[3,1,2]),p,q,1),3);
    end
end

end



