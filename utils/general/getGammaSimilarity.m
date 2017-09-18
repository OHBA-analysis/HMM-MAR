function [S,assig, gamma1] = getGammaSimilarity (gamma1, gamma2)
% Computes a measure of similarity between two sets of state time courses.
% These can have different number of states, but they must have the same
% number of time points. 
% S: similarity, measured as the sum of overlapping probabilities under the
%       optimal state alignment
% assig: optimal state aligmnent (uses munkres' algorithm)
% gamma1: the first set of state time courses reordered to match gamma2
%
% Author: Diego Vidaurre, University of Oxford (2017)

[T, K] = size(gamma1);
K2 = size(gamma2,2);

if K<K2
    gamma1 = [gamma1 zeros(T,K2-K)];
    K = K2;
else
    if K>K2
        gamma2 = [gamma2 zeros(T,K-K2)];
    end
end

M = zeros(K,K); % minus similarities
for i=1:K 
    for j=1:K
        M(i,j) = - sum(gamma1(:,i) .* gamma2(:,j));
    end
end 

[assig,cost] = munkres(M);
S = - cost / T;

if nargout > 2
    gamma1(:,assig) = gamma1; 
end

end