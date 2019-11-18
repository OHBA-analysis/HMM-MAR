function L = obslikebinomial(X,hmm)
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
    % expectation of log(p) is psi(a) - psi(a+b):
    pterm = [X .* repmat( psi(hmm.state(k).W.a) - psi(hmm.state(k).W.a + hmm.state(k).W.b) ,T,1)];
    qterm = [(~X) .* repmat(psi(hmm.state(k).W.b) - psi(hmm.state(k).W.a + hmm.state(k).W.b) ,T,1)];
    L(:,k) = sum(pterm+qterm,2);
end

%indices of coefficients:
S = hmm.train.S==1;
Sind = hmm.train.S==1; 
%setstateoptions;

L=exp(L);
end
