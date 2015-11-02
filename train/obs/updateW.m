function [hmm,XW] = updateW(hmm,Gamma,residuals,XX,XXGXX)

K = length(hmm.state); ndim = hmm.train.ndim;
XW = zeros(K,size(XX{1},1),ndim);

for k=1:K
    setstateoptions;
    if isempty(orders) && train.zeromean, continue; end
    if strcmp(train.covtype,'diag') || strcmp(train.covtype,'full'), omega = hmm.state(k).Omega;
    else omega = hmm.Omega;
    end
    if train.uniqueAR || ndim==1 % it is assumed that order>0 and cov matrix is diagonal
        XY = zeros(length(orders),1);
        XGX = zeros(length(orders));
        for n=1:ndim
            ind = n:ndim:size(XX{kk},2);
            iomegan = omega.Gam_shape / omega.Gam_rate(n);
            XGX = XGX + iomegan * XXGXX{k}(ind,ind);
            XY = XY + (iomegan * XX{kk}(:,ind)' .* repmat(Gamma(:,k)',length(ind),1)) * residuals(:,n);
        end
        if ~isempty(train.prior)
            hmm.state(k).W.S_W = inv(train.prior.iS + XGX);
            hmm.state(k).W.Mu_W = hmm.state(k).W.S_W * (XY + train.prior.iSMu); % order by 1
        else
            if train.zeromean==0,
                regterm = diag([hmm.state(k).prior.Mean.iS (hmm.state(k).alpha.Gam_shape ./ ...
                    hmm.state(k).alpha.Gam_rate) ]);
            else
                regterm = diag((hmm.state(k).alpha.Gam_shape ./  hmm.state(k).alpha.Gam_rate));
            end
            hmm.state(k).W.S_W = inv(regterm + XGX);
            hmm.state(k).W.Mu_W = hmm.state(k).W.S_W * XY; % order by 1
        end        
        for n=1:ndim
            ind = n:ndim:size(XX{kk},2);
            XW(k,:,n) = XX{kk}(:,ind) * hmm.state(k).W.Mu_W;
        end
        
    elseif strcmp(train.covtype,'diag') || strcmp(train.covtype,'uniquediag')
        for n=1:ndim
            ndim_n = sum(S(:,n));
            if ndim_n==0 && train.zeromean==1, continue; end
            regterm = [];
            if ~train.zeromean, regterm = hmm.state(k).prior.Mean.iS(n); end
            if ~isempty(orders)
                alphaterm = repmat( (hmm.state(k).alpha.Gam_shape ./  hmm.state(k).alpha.Gam_rate), ndim_n, 1);
                if ndim>1
                    regterm = [regterm; repmat(hmm.state(k).sigma.Gam_shape(S(:,n),n) ./ ...
                        hmm.state(k).sigma.Gam_rate(S(:,n),n), length(orders), 1).*alphaterm(:) ];
                else
                    regterm = [regterm; alphaterm(:)];
                end
            end
            if isempty(regterm), regterm = 0; end
            regterm = diag(regterm);
            prec =  regterm + (omega.Gam_shape / omega.Gam_rate(n)) * XXGXX{k}(Sind(:,n),Sind(:,n));
            hmm.state(k).W.S_W(n,Sind(:,n),Sind(:,n)) = inv(prec);
            hmm.state(k).W.Mu_W(Sind(:,n),n) = (( permute(hmm.state(k).W.S_W(n,Sind(:,n),Sind(:,n)),[2 3 1]) * ...
                (omega.Gam_shape / omega.Gam_rate(n)) * XX{kk}(:,Sind(:,n))') .* ...
                repmat(Gamma(:,k)',sum(Sind(:,n)),1)) * residuals(:,n);
        end;
        XW(k,:,:) = XX{kk} * hmm.state(k).W.Mu_W;
        
    else % this only works if all(S(:)==1);  any(S(:)==0) is just not yet implemented 
        mlW = (( XXGXX{k} \ XX{kk}') .* repmat(Gamma(:,k)',...
            (~train.zeromean)+ndim*length(orders),1) * residuals)';
        regterm = [];
        if ~train.zeromean, regterm = hmm.state(k).prior.Mean.iS; end
        if ~isempty(orders) 
            sigmaterm = (hmm.state(k).sigma.Gam_shape(S) ./ hmm.state(k).sigma.Gam_rate(S));
            sigmaterm = repmat(sigmaterm, length(orders), 1);
            alphaterm = repmat( (hmm.state(k).alpha.Gam_shape ./  hmm.state(k).alpha.Gam_rate), sum(S(:)), 1);
            alphaterm = alphaterm(:);
            regterm = [regterm; (alphaterm .* sigmaterm)];
        end
        if isempty(regterm), regterm = 0; end
        regterm = diag(regterm);
        prec = omega.Gam_shape * omega.Gam_irate;
        gram = kron(XXGXX{k}, prec);
        hmm.state(k).W.S_W = inv(regterm + gram);
        muW = hmm.state(k).W.S_W * gram * mlW(:);
        hmm.state(k).W.Mu_W = reshape(muW,ndim,~train.zeromean+ndim*length(orders))';
        XW(k,:,:) = XX{kk} * hmm.state(k).W.Mu_W;
    end
    
end

end