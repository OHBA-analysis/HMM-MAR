function avLL = GammaavLL(hmm,Gamma,Xi,T)
% average loglikelihood for state time course
if isfield(hmm.train,'grouping')
    Q = length(unique(hmm.train.grouping));
else
    Q = 1;
end
N = length(T); 
order = hmm.train.maxorder;
if Q>1, Dir_alpha = hmm.Dir_alpha';
else Dir_alpha = hmm.Dir_alpha;
end
avLL = 0; K = size(Gamma,2);
jj = zeros(N,1); % reference to first time point of the segments
for in = 1:N
    jj(in) = sum(T(1:in-1)) - order*(in-1) + 1;
end
% avLL initial state
if Q>1
    for i = 1:Q
        PsiDir_alphasum = psi(sum(Dir_alpha(:,i)));
        ii = hmm.train.grouping==i;
        for l = 1:K
            avLL = avLL + sum(Gamma(jj(ii),l)) * (psi(hmm.Dir_alpha(l,i)) - PsiDir_alphasum);
        end
    end
else
    PsiDir_alphasum = psi(sum(Dir_alpha));
    for l = 1:K 
        avLL = avLL + sum(Gamma(jj,l)) * (psi(hmm.Dir_alpha(l)) - PsiDir_alphasum);
    end
end
% avLL remaining time points
for i = 1:Q
    if Q > 1
        ii = find(hmm.train.grouping==i)';
    else
        ii = 1:length(T);
    end
    for k = 1:K
        PsiDir2d_alphasum=psi(sum(hmm.Dir2d_alpha(:,k,i)));
        for l = 1:K
            if Q==1
                avLL = avLL + sum(Xi(:,l,k)) * (psi(hmm.Dir2d_alpha(l,k))-PsiDir2d_alphasum);
            else
                for n = ii
                    t = (1:T(n)-1-order) + sum(T(1:n-1)) - (order+1)*(n-1) ;
                    avLL = avLL + sum(Xi(t,l,k)) * (psi(hmm.Dir2d_alpha(l,k,i))-PsiDir2d_alphasum);
                end
            end
        end
    end
end

end
