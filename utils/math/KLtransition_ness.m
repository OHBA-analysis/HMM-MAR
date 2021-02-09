function KLdiv = KLtransition_ness(ness)
KLdiv = 0;
K = ness.K;
for k = 1:K
    % KL divergence for the transition and initial probabilities is zero
    %   because the first state is always OFF
    % KL-divergence for transition prob
    if ~isfield(ness.train,'id_mixture') || ~ness.train.id_mixture % proper HMM?
        for k2 = 1:2
            KLdiv = KLdiv + dirichlet_kl(ness.state(k).Dir2d_alpha(k2,:),...
                ness.state(k).prior.Dir2d_alpha(k2,:));
        end
    end
    if isnan(KLdiv)
        error(['Error computing kullback-leibler divergence of the ' ...
            'transition prob matrix - Please report the error'])
    end
end
end