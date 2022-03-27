function [hmm,Gamma,Xi] = hmmprune(hmm,Gamma,Xi,threshold)

K = hmm.K;
if nargin < 3, Xi = []; end
if nargin < 4, threshold = 5.0; end % in time points

Gammasum = sum(Gamma);

is_active = false(1,K); % length = to the last no. of states (=the original K if dropstates==0)
for k = 1:K
    is_active(k) = Gammasum(:,k) >= threshold;
end

Gamma = Gamma(:,is_active);
Gamma = bsxfun(@rdivide,Gamma,sum(Gamma,2));
hmm.state = hmm.state(is_active);

hmm.K = sum(is_active);
hmm.prior.Dir_alpha = hmm.prior.Dir_alpha(is_active);
hmm.prior.Dir2d_alpha = hmm.prior.Dir2d_alpha(is_active,is_active);
hmm.Dir2d_alpha = hmm.Dir2d_alpha(is_active,is_active,:);
hmm.P = hmm.P(is_active,is_active,:);
hmm.Dir_alpha = hmm.Dir_alpha(is_active);
hmm.Pi = hmm.Pi(is_active);

hmm.train.Pstructure = hmm.train.Pstructure(is_active,is_active);
hmm.train.Pistructure = hmm.train.Pistructure(is_active);

if ~isempty(Xi)
    Xi = Xi(:,is_active,is_active);
end
hmm.train.active = ones(1,sum(is_active));
% Renormalize
hmm.P = bsxfun(@rdivide,hmm.P,sum(hmm.P,2));
hmm.Pi = hmm.Pi ./ sum(hmm.Pi);


end
