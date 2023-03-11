function [Xstar,Tstar] = embeddata_batched(X,T,embeddedlags)

tp_less = max(embeddedlags) + max(-embeddedlags);
binsize = length(embeddedlags);

p = size(X,2);
N = length(T);

Tstar = T - tp_less;
Xstar = zeros(sum(Tstar),binsize,p);

for j = 1:N
    ind1 = (1:T(j)) + sum(T(1:j-1));
    ind2 = (1:Tstar(j)) + sum(Tstar(1:j-1));
    for i = 1:p
        Xstar(ind2,:,i) = embedx(X(ind1,i),embeddedlags);
    end
end

end