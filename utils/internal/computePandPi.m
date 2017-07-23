function [P,Pi] = computePandPi(Dir_alpha,Dir2d_alpha)

K = size(Dir2d_alpha,2);
if length(size(Dir2d_alpha))==3
    Q = length(size(Dir2d_alpha));
else
    Q = 1;
end
P = zeros(K,K,Q);
if Q==1, Pi = zeros(1,K); 
else, Pi = zeros(K,Q);
end
for i = 1:Q
    for j = 1:K
        PsiSum = psi(sum(Dir2d_alpha(j,:,i)));
        for k = 1:K
            P(j,k,i) = exp(psi(Dir2d_alpha(j,k,i))-PsiSum);
        end
        P(j,:,i) = P(j,:,i) ./ sum(P(j,:,i));
    end
    if Q==1
        PsiSum = psi(sum(Dir_alpha));
        for k = 1:K
            Pi(k) = exp(psi(Dir_alpha(k))-PsiSum);
        end
        Pi = Pi ./ sum(Pi);
    else
        PsiSum = psi(sum(Dir_alpha(:,i)));
        for k = 1:K
            Pi(k,i) = exp(psi(Dir_alpha(k,i))-PsiSum);
        end
        Pi(:,i) = Pi(:,i) ./ sum(Pi(:,i));        
    end
end

end