function L = obslikepoisson(X,hmm)
%
% Evaluate likelihood of data given observation model, for one continuous trial
%
% INPUT
% X          N by ndim data matrix
% hmm        hmm data structure
%
% OUTPUT
% B          Likelihood of N data points
%
% Author: Cam Higgins, OHBA, University of Oxford

K = hmm.K;
[T,ndim] = size(X);

L = zeros(T,K);  

for k=1:K
    constterm = -gammaln(X + 1);
    % note the expectation of log(lambda) is -log(lambda_b) + psigamma(lambda_a)
    num = (X.*(repmat(psi(hmm.state(k).W.W_shape),T,1) - log(hmm.state(k).W.W_rate))) - ...
        repmat(hmm.state(k).W.W_mean,T,1);
    L(:,k) = sum(num + constterm,2);
end
L = exp(L);
end
