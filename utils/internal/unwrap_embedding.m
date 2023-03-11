function [X,T] = unwrap_embedding(Xstar,Tstar,embeddedlags)

tp_less = max(embeddedlags) + max(-embeddedlags);

p = size(Xstar,2);
N = length(Tstar);

T = Tstar + tp_less;
X = zeros(sum(T),p);

for j = 1:N
    ind1 = (1:T(j)) + sum(T(1:j-1));
    ind2 = (1:Tstar(j)) + sum(Tstar(1:j-1));
    Xj = zeros(T(j),p);
    for l = 1:length(embeddedlags)
        Xj((1:Tstar(j))+(l-1),:) = Xj((1:Tstar(j))+(l-1),:) + Xstar(ind2,:);
    end
    X(ind1,:) = Xj;
end

end