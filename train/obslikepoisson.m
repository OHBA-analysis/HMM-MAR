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


% not familiar with caching commands so omitting for now
% if nargin < 5 || isempty(cache) 
%     use_cache = false;
% else
%     use_cache = true;
% end

K=hmm.K;
[T,ndim]=size(X);

L = zeros(T,K);  

for k=1:K
    constterm = -gammaln(X+1);
    % note the expectation of log(lambda) is -log(lambda_b) + psigamma(lambda_a)
    num = (X.*(repmat(psi(hmm.state(k).W.W_shape),T,1)-log(hmm.state(k).W.W_rate)))- hmm.state(k).W.W_mean;
    L(:,k) = sum(num+constterm,2);
end

%indices of coefficients:
S = hmm.train.S==1;
Sind = hmm.train.S==1; 
%setstateoptions;

%for Y with >2 dimensions, change here!!!
%n=Xdim+hmm.train.logisticYdim;

L=exp(L);
end
