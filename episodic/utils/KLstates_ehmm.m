function KLdiv = KLstates_ehmm(ehmm)

ndim = length(ehmm.Omega.Gam_rate);
K = ehmm.K; 
setstateoptions;
np = length(ehmm.state_shared(1).Mu_W);

OmegaKL = 0;
for n = 1:ndim
    if ~regressed(n), continue; end
    OmegaKL = OmegaKL + gamma_kl(ehmm.Omega.Gam_shape,ehmm.prior.Omega.Gam_shape, ...
        ehmm.Omega.Gam_rate(n),ehmm.prior.Omega.Gam_rate(n));
end

WKL = 0;
for n = 1:ndim
    prior_prec = [];
    for k = 1:K+1
        hs = ehmm.state(k);
        if ~regressed(n), continue; end
        if train.zeromean==0
            prior_prec = hs.prior.Mean.iS(n);
        end
        if train.order > 0
            ndim_n = sum(S(:,n));
            alphaterm = repmat( (hs.alpha.Gam_shape ./  hs.alpha.Gam_rate), ndim_n, 1);
            sigmaterm = repmat(hs.sigma.Gam_shape(S(:,n),n) ./ ...
                hs.sigma.Gam_rate(S(:,n),n), length(orders), 1);
            prior_prec = [prior_prec; sigmaterm .* alphaterm(:)] ;
        end
    end
    prior_var = diag(1 ./ prior_prec);
    WKL = WKL + gauss_kl(ehmm.state_shared(n).Mu_W,zeros(np,1), ...
        ehmm.state_shared(n).S_W, prior_var);
end

sigmaKL = 0; alphaKL = 0;
for k = 1:K+1
    hs = ehmm.state(k);
    pr = ehmm.state(k).prior;
    if isempty(train.prior) && ~isempty(orders) && ~train.uniqueAR && ndim>1
        for n1 = 1:ndim
            for n2 = 1:ndim
                if (train.symmetricprior && n2<n1) || S(n1,n2)==0, continue; end
                sigmaKL = sigmaKL + gamma_kl(hs.sigma.Gam_shape(n1,n2),pr.sigma.Gam_shape(n1,n2), ...
                    hs.sigma.Gam_rate(n1,n2),pr.sigma.Gam_rate(n1,n2));
            end
        end
    end
    if isempty(train.prior) &&  ~isempty(orders)
        for i = 1:length(orders)
            alphaKL = alphaKL + gamma_kl(hs.alpha.Gam_shape,pr.alpha.Gam_shape, ...
                hs.alpha.Gam_rate(i),pr.alpha.Gam_rate(i));
        end
    end
end

KLdiv = [OmegaKL WKL sigmaKL alphaKL];

end
