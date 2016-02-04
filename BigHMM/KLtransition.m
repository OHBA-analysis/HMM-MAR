function KLdiv = KLtransition(markovTrans)
% KL divergence for the transition and initial probabilities
KLdiv = dirichlet_kl(markovTrans.Dir_alpha,markovTrans.prior.Dir_alpha); % + ...
%    dirichlet_kl(hmm.Dir2d_alpha(:)',hmm.prior.Dir2d_alpha(:)'); 
K = length(markovTrans.Dir_alpha);
for l=1:K,
    % KL-divergence for transition prob
    KLdiv = KLdiv + dirichlet_kl(markovTrans.Dir2d_alpha(l,:),markovTrans.prior.Dir2d_alpha(l,:));
end
end