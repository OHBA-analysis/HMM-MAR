function KLdiv = KLtransition(hmm)
% KL divergence for the transition and initial probabilities
KLdiv = dirichlet_kl(hmm.Dir_alpha,hmm.prior.Dir_alpha); % + ...
%    dirichlet_kl(hmm.Dir2d_alpha(:)',hmm.prior.Dir2d_alpha(:)'); 
K = length(hmm.Dir_alpha);
for l=1:K,
    % KL-divergence for transition prob
    KLdiv = KLdiv + dirichlet_kl(hmm.Dir2d_alpha(l,:),hmm.prior.Dir2d_alpha(l,:));
end
end