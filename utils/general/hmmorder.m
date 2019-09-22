function [hmm,Gamma,order] = hmmorder(hmm,Gamma,T,indexes)
% Reorder the states according to their mean ocurrence
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2019)

exponent = 5; % this is to boost the main peak
K = size(Gamma,2);
mGamma = getFractionalOccupancy (Gamma,T,[],1);
if nargin>3, mGamma = mGamma(indexes,:); end 
FO = mean(mGamma); not_used = (FO < 0.01);
mGamma = mGamma.^exponent;
mGamma = mGamma ./ repmat(sum(mGamma),size(mGamma,1),1);
t = linspace(-1,1,size(mGamma,1));
r = sum(mGamma .* repmat(t',1,K));
r(not_used) = Inf; 
[~,order] = sort(r);
Gamma = Gamma(:,order);
hmm.state = hmm.state(order);
hmm.P = hmm.P(order,order);
hmm.Pi = hmm.Pi(order);
hmm.Dir2d_alpha = hmm.Dir2d_alpha(order,order);
hmm.Dir_alpha = hmm.Dir_alpha(order);
hmm.prior.Dir2d_alpha = hmm.prior.Dir2d_alpha(order,order);
hmm.prior.Dir_alpha = hmm.prior.Dir_alpha(order);
hmm.train.Pstructure = hmm.train.Pstructure(order,order);
hmm.train.Pistructure = hmm.train.Pistructure(order);
hmm.train.active = hmm.train.active(order);

end