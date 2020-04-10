function t = iseuclidean(D)
% D is (1 by pairs)
m = size(D,2);
% make sure it's a valid dissimilarity matrix
n = ceil(sqrt(2*m)); % (1+sqrt(1+8*m))/2, but works for large m
if n*(n-1)/2 == m && all(D >= 0)
    D = squareform(D);
else
    warning(message('stats:iseuclidean:NotDistanceMatrix'))
    t = false;
    return
end
P = eye(n) - repmat(1/n,n,n);
B = P * (-.5 * D .* D) * P;
g = eig((B+B')./2); % guard against spurious complex e-vals from roundoff
t = all(-eps(class(g))^(3/4) * max(abs(g)) <= g); 
% all non-negative eigenvals (within roundoff)?
end