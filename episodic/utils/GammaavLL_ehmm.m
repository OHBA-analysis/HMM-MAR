function avLL = GammaavLL_ehmm(ehmm,Xi)
% average loglikelihood for state time course

K = ehmm.K;
avLL = 0; 

for k = 1:K
    % first time point is always OFF so it doesn't add
    PsiDir2d_alphasum = zeros(2,1);
    for l = 1:2, PsiDir2d_alphasum(l) = psi(sum(ehmm.state(k).Dir2d_alpha(l,:))); end
    for l1 = 1:2
        for l2 = 1:2
            avLL = avLL + sum(Xi(:,k,l2,l1)) * ...
                (psi(ehmm.state(k).Dir2d_alpha(l2,l1))-PsiDir2d_alphasum(l2));
            if isnan(avLL)
                error(['Error computing log likelihood of the state time courses  - ' ...
                    'Out of precision?'])
            end
        end
    end
end

end
