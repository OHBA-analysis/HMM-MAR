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
        kk = hmm.train.Pistructure;
        KLdiv = KLdiv + dirichlet_kl(hmm.Dir_alpha(kk,i)',hmm.prior.Dir_alpha(kk));
    end
else 
    if ~all(hmm.Dir_alpha==hmm.prior.Dir_alpha)
        KLdiv = KLdiv + dirichlet_kl(hmm.Dir_alpha,hmm.prior.Dir_alpha);
    end
end
% KL-divergence for transition prob
if ~isfield(hmm.train,'id_mixture') || ~hmm.train.id_mixture % proper HMM?
    K = length(hmm.state);
    for i = 1:Q
        for k = 1:K
            kk = hmm.train.Pstructure(k,:);
            KLdiv = KLdiv + dirichlet_kl(hmm.Dir2d_alpha(k,kk,i),hmm.prior.Dir2d_alpha(k,kk));
        end
    end
end
if isnan(KLdiv)
    error(['Error computing kullback-leibler divergence of the transition prob matrix - ' ...
        'Please report the error'])
end
end