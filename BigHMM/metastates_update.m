function metastates = metastates_update(metastates,metastate_noisy,rho,update)
% update==1, W; update==2, Omega; update==3, sigma; update==4, alpha
K = length(metastate_noisy.state);
for k = 1:K
    if update==1 && isfield(metastate_noisy.state(1),'W') && ~isempty(metastate_noisy.state(1).W.Mu_W)
        metastates.state(k).W.Mu_W = (1-rho) * metastates.state(k).W.Mu_W + ...
            rho * metastate_noisy.state(k).W.Mu_W;
        metastates.state(k).W.iS_W = (1-rho) * metastates.state(k).W.iS_W + ...
            rho * metastate_noisy.state(k).W.iS_W;
        if strcmp(metastate_noisy.train.covtype,'full') || strcmp(metastate_noisy.train.covtype,'uniquefull')
            metastates.state(k).W.S_W = inv(metastates.state(k).W.iS_W);
        else
            for n = 1:size(metastates.state(k).W.S_W,1)
                metastates.state(k).W.S_W(n,:,:) = inv(permute(metastates.state(k).W.iS_W(n,:,:),[2 3 1]));
            end
        end
    elseif update==2 && isfield(metastate_noisy.state(1),'Omega')
        metastates.state(k).Omega.Gam_rate = (1-rho) * metastates.state(k).Omega.Gam_rate + ...
            rho * metastate_noisy.state(k).Omega.Gam_rate;
        metastates.state(k).Omega.Gam_shape = (1-rho) * metastates.state(k).Omega.Gam_shape + ...
            rho * metastate_noisy.state(k).Omega.Gam_shape;
        if strcmp(metastate_noisy.train.covtype,'full') || strcmp(metastate_noisy.train.covtype,'uniquefull')
            metastates.state(k).Omega.Gam_irate = inv(metastates.state(k).Omega.Gam_rate);
        else
            metastates.state(k).Omega.Gam_irate = 1 ./ metastates.state(k).Omega.Gam_rate;
        end
    elseif update==3 && isfield(metastate_noisy.state(1),'sigma')
        metastates.state(k).sigma.Gam_rate = (1-rho) * metastates.state(k).sigma.Gam_rate + ...
            rho * metastate_noisy.state(k).sigma.Gam_rate;
        metastates.state(k).sigma.Gam_shape = (1-rho) * metastates.state(k).sigma.Gam_shape + ...
            rho * metastate_noisy.state(k).sigma.Gam_shape;
    elseif update==4 && isfield(metastate_noisy.state(1),'alpha')
        metastates.state(k).alpha.Gam_rate = (1-rho) * metastates.state(k).alpha.Gam_rate + ...
            rho * metastate_noisy.state(k).alpha.Gam_rate;
        metastates.state(k).alpha.Gam_shape = (1-rho) * metastates.state(k).alpha.Gam_shape + ...
            rho * metastate_noisy.state(k).alpha.Gam_shape;
    end
end
end