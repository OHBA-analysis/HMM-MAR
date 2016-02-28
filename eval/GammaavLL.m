function avLL = GammaavLL(hmm,Gamma,Xi,T)
% average loglikelihood for state time course
avLL = 0; K = size(Gamma,2);
jj = zeros(length(T),1); % reference to first time point of the segments
for in=1:length(T);
    jj(in) = sum(T(1:in-1)) - hmm.train.maxorder*(in-1) + 1;
end
PsiDir_alphasum = psi(sum(hmm.Dir_alpha,2));
for l=1:K,
    % avLL initial state  
    avLL = avLL + sum(Gamma(jj,l)) * (psi(hmm.Dir_alpha(l)) - PsiDir_alphasum);
end     
% avLL remaining time points  
for k=1:K,
    PsiDir2d_alphasum=psi(sum(hmm.Dir2d_alpha(:,k)));
    for l=1:K,
        avLL = avLL + sum(Xi(:,l,k)) * (psi(hmm.Dir2d_alpha(l,k))-PsiDir2d_alphasum);
    end
end
end
