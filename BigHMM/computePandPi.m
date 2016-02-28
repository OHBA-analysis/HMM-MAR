function [P,Pi] = computePandPi(Dir_alpha,Dir2d_alpha)
K = length(Dir_alpha);
P = zeros(K); Pi = zeros(1,K);
for j = 1:K
    PsiSum = psi(sum(Dir2d_alpha(j,:)));
    for i = 1:K,
        P(j,i) = exp(psi(Dir2d_alpha(j,i))-PsiSum);
    end;
    P(j,:) = P(j,:) ./ sum(P(j,:));
end
PsiSum = psi(sum(Dir_alpha,2));
for i = 1:K
    Pi(i) = exp(psi(Dir_alpha(i))-PsiSum);
end
Pi = Pi ./ sum(Pi);
end