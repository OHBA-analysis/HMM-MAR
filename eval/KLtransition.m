function KLdiv = KLtransition(hmm)
KLdiv = 0; 
if length(size(hmm.Dir2d_alpha))==3
    Q = size(hmm.Dir2d_alpha,3);
else
    Q = 1;
end
% KL divergence for the transition and initial probabilities
if Q > 1
    for i = 1:Q
        KLdiv = KLdiv + dirichlet_kl(hmm.Dir_alpha(:,i)',hmm.prior.Dir_alpha);
    end
else 
    KLdiv = KLdiv + dirichlet_kl(hmm.Dir_alpha,hmm.prior.Dir_alpha);
end
% KL-divergence for transition prob
K = size(hmm.Dir_alpha,1);
for i = 1:Q
    for k = 1:K
        KLdiv = KLdiv + dirichlet_kl(hmm.Dir2d_alpha(k,:,i),hmm.prior.Dir2d_alpha(k,:));
    end
end

end