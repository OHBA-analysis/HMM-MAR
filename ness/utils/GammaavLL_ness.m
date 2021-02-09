function avLL = GammaavLL_ness(hmm,Xi)
% average loglikelihood for state time course

K = hmm.K;
avLL = 0; 

% ON states
for k = 1:K
    % first time point is always OFF so it doesn't add
    % avLL remaining time points
    PsiDir2d_alphasum = zeros(2,1);
    for l = 1:2, PsiDir2d_alphasum(l) = psi(sum(hmm.state(k).Dir2d_alpha(l,:))); end
    for l1 = 1:2
        for l2 = 1:2
            avLL = avLL + sum(Xi(:,k,l2,l1)) * ...
                (psi(hmm.state(k).Dir2d_alpha(l2,l1))-PsiDir2d_alphasum(l2));
            if isnan(avLL)
                error(['Error computing log likelihood of the state time courses  - ' ...
                    'Out of precision?'])
            end
        end
    end
end
% % baseline
% Dir2d_alpha = 0; Dir2d_alpha_rest = zeros(1,K);
% for k = 1:K
%     Dir2d_alpha = Dir2d_alpha + hmm.state(k).Dir2d_alpha(2,2);
%     Dir2d_alpha_rest(k) = hmm.state(k).Dir2d_alpha(2,1);
% end
% Dir2d_alpha = [Dir2d_alpha Dir2d_alpha_rest];
% PsiDir2d_alphasum = psi(sum(Dir2d_alpha));
% avLL = avLL + sum(sum(Xi(:,:,2,2),2)) * (psi(Dir2d_alpha(1))-PsiDir2d_alphasum);
% for k = 1:K
%     avLL = avLL + sum(Xi(:,k,2,1)) * (psi(Dir2d_alpha(k+1))-PsiDir2d_alphasum);
% end

end
