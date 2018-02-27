function beta = tudabeta(tuda)
% Given a precomputed TUDA model, it returns the decoding coefficients
% (aka the "beta" parameters) 
%
% INPUT
% tuda: Estimated TUDA model, using tudatrain
%
% OUTPUT
% beta: (no. states by no. channels by no. predicted features) array of
%           decoding coefficients
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2018)

q = size(tuda.train.S,1) - find(tuda.train.S(1,:)>0,1) + 1;
p = find(tuda.train.S(1,:)>0,1) - 1; 
beta = zeros(tuda.K,p,q);
for k = 1:tuda.K
    beta(k,:,:) = tuda.state(k).W.Mu_W(1:p,p+1:end);
end

end



