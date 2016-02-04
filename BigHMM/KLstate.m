function KLdiv = KLstate(metastate,prior,options) 
% KL divergence between a state and its prior (cov and mean)
KLdiv = 0;  
orders = options.orders;
% cov matrix
if strcmp(options.covtype,'full') 
    ndim = size(metastate.Omega.Gam_rate,1);
    KLdiv = wishart_kl(metastate.Omega.Gam_rate,prior.Omega.Gam_rate, ...
                metastate.Omega.Gam_shape,prior.Omega.Gam_shape);
elseif strcmp(options.covtype,'diag')
    ndim = length(metastate.Omega.Gam_rate);
    for n=1:ndim
        KLdiv = KLdiv + gamma_kl(metastate.Omega.Gam_shape,prior.Omega.Gam_shape, ...
            metastate.Omega.Gam_rate(n),prior.Omega.Gam_rate(n));
    end
end
% W
if ~options.zeromean || ~isempty(options.orders)
    if strcmp(options.covtype,'full') 
        prior_prec = [];
        if options.zeromean==0
            prior_prec = prior.Mean.iS;
        end
        if ~isempty(orders)
            sigmaterm = (metastate.sigma.Gam_shape ./ metastate.sigma.Gam_rate );
            sigmaterm = repmat(sigmaterm, length(orders), 1);
            alphaterm = repmat( (metastate.alpha.Gam_shape ./  metastate.alpha.Gam_rate), ndim^2, 1);
            alphaterm = alphaterm(:);
            prior_prec = [prior_prec; alphaterm .* sigmaterm];
        end
        prior_var = diag(1 ./ prior_prec);
        mu_w = metastate.W.Mu_W';
        mu_w = mu_w(:);        
        KLdiv = KLdiv + gauss_kl(mu_w,zeros(length(mu_w),1), metastate.W.S_W, prior_var); 
    else
        
        for n=1:ndim
            prior_prec = [];
            if options.zeromean==0
                prior_prec = metastate.prior.Mean.iS(n);
            end
            if ~isempty(orders)
                alphamat = repmat( (metastate.alpha.Gam_shape ./  metastate.alpha.Gam_rate), ndim, 1);
                prior_prec = [prior_prec; repmat(metastate.sigma.Gam_shape(:,n) ./ ...
                    metastate.sigma.Gam_rate(:,n), length(orders), 1) .* alphamat(:)] ;
            end
            prior_var = diag(1 ./ prior_prec);
            KLdiv = KLdiv + gauss_kl(metastate.W.Mu_W(:,n),zeros(ndim,1), ...
                permute(metastate.W.S_W(n,:,:),[2 3 1]), prior_var);
        end
    end
end
% sigma and alpha
if ~isempty(orders)
    for n1=1:ndim % sigma
        for n2=1:ndim
            KLdiv = KLdiv + gamma_kl(metastate.sigma.Gam_shape(n1,n2),prior.sigma.Gam_shape(n1,n2), ...
                metastate.sigma.Gam_rate(n1,n2),prior.sigma.Gam_rate(n1,n2));
        end
    end
    for i=1:length(orders)
        KLdiv = KLdiv + gamma_kl(metastate.alpha.Gam_shape,prior.alpha.Gam_shape, ...
            metastate.alpha.Gam_rate(i),pr.alpha.Gam_rate(i));
    end
end
end
