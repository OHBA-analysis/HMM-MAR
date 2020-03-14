function [S,assig, gamma1] = getGammaSimilarity (gamma1, gamma2)
% Computes a measure of similarity between two sets of state time courses.
% These can have different number of states, but they must have the same
% number of time points. 
% If gamma2 is a cell, then it aggregates the similarity measures across
% elements of gamma2 
% S: similarity, measured as the sum of joint probabilities under the
%       optimal state alignment
% assig: optimal state aligmnent (uses munkres' algorithm)
% gamma1: the first set of state time courses reordered to match gamma2
%
% Author: Diego Vidaurre, University of Oxford (2017)

if iscell(gamma2), N = length(gamma2); 
else, N = 1; 
end

[T, K] = size(gamma1);

gamma1_0 = gamma1; 

M = zeros(K,K); % cost

for j = 1:N
    
    if iscell(gamma2), g = gamma2{j};
    else g = gamma2;
    end
    
    K2 = size(g,2);
    
    if K < K2
        gamma1 = [gamma1_0 zeros(T,K2-K)];
        K = K2;
    elseif K>K2
        g = [g zeros(T,K-K2)];
    end
    
    for k1 = 1:K
        for k2 = 1:K
            M(k1,k2) = M(k1,k2) + (T - sum(min(gamma1(:,k1), g(:,k2)))) / T / N;
        end
    end
    
end

[assig,cost] = munkres(M);
S = K - cost;

if nargout > 2
    gamma1(:,assig) = gamma1;
end

end