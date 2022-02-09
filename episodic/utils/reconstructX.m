function X = reconstructX(XX,T,order)
ndim = size(XX,2) / order; 
X = zeros(sum(T),ndim); N = length(T); 
for j = 1:N
    t0 = sum(T(1:j-1)); t02 = t0 - (j-1)*order;
    ind = (1:T(j)-order); 
    X(t0 + ind,:) = XX(t02 + ind, (ndim * (order-1)) + (1:ndim));
    for i = order-1:-1:1
        ii = order - i;
        X(t0 + T(j)-order + ii,:) = ...
            XX(t02 + T(j)-order, (ndim * (i-1)) + (1:ndim));
    end
end
end