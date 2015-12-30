function hmm = updateOmega(hmm,Gamma,Gammasum,residuals,Tres,XX,XXGXX,XW)

K = length(hmm.state); ndim = hmm.train.ndim;
S = hmm.train.S==1; regressed = sum(S,1)>0;

if strcmp(hmm.train.covtype,'uniquediag') && hmm.train.uniqueAR 
    % all are AR and there's a single covariance matrix
    hmm.Omega.Gam_rate = hmm.prior.Omega.Gam_rate;
    for k=1:K
        if hmm.train.multipleConf, kk = k;
        else kk = 1;
        end
        XWk = permute(XW(k,:,:),[2 3 1]);
        e = (residuals - XWk).^2;
        swx2 = zeros(Tres,ndim);
        for n=1:ndim
            ind = n:ndim:size(XX{kk},2);
            tmp = XX{kk}(:,ind) * hmm.state(k).W.S_W;
            swx2(:,n) = sum(tmp .* XX{kk}(:,ind),2);
        end;
        hmm.Omega.Gam_rate = hmm.Omega.Gam_rate + ...
            0.5 * sum( repmat(Gamma(:,k),1,ndim) .* (e + swx2) );
    end;
    hmm.Omega.Gam_shape = hmm.prior.Omega.Gam_shape + Tres / 2;    
    
elseif strcmp(hmm.train.covtype,'uniquediag')
    hmm.Omega.Gam_rate(regressed) = hmm.prior.Omega.Gam_rate(regressed);
    for k=1:K
        setstateoptions;
        XWk = permute(XW(k,:,:),[2 3 1]);
        e = (residuals(:,regressed) - XWk(:,regressed)).^2;
        swx2 = zeros(Tres,ndim);
        if ~isempty(hmm.state(k).W.Mu_W)
            for n=1:ndim
                if ~regressed(n), continue; end
                if ndim==1
                    tmp = XX{kk}(:,Sind(:,n)) * hmm.state(k).W.S_W;
                else
                    tmp = XX{kk}(:,Sind(:,n)) * permute(hmm.state(k).W.S_W(n,Sind(:,n),Sind(:,n)),[2 3 1]);
                end
                swx2(:,n) = sum(tmp .* XX{kk}(:,Sind(:,n)),2);
            end;
        end
        hmm.Omega.Gam_rate(regressed) = hmm.Omega.Gam_rate(regressed) + ...
            0.5 * sum( repmat(Gamma(:,k),1,sum(regressed)) .* (e + swx2(:,regressed) ) );
    end;
    hmm.Omega.Gam_shape = hmm.prior.Omega.Gam_shape + Tres / 2;
    
elseif strcmp(hmm.train.covtype,'uniquefull')
    hmm.Omega.Gam_rate(regressed,regressed) = hmm.prior.Omega.Gam_rate(regressed,regressed);
    for k=1:K
        setstateoptions;
        XWk = permute(XW(k,:,:),[2 3 1]);
        e = (residuals(:,regressed) - XWk(:,regressed));
        e = (e' .* repmat(Gamma(:,k)',sum(regressed),1)) * e;
        swx2 =  zeros(ndim,ndim);
        if ~isempty(hmm.state(k).W.Mu_W)  
            for n1=find(regressed)
                for n2=find(regressed)
                    if n2<n1, continue, end;
                    index1 = (0:length(orders)*ndim+(~hmm.train.zeromean)-1) * ndim + n1;
                    index2 = (0:length(orders)*ndim+(~hmm.train.zeromean)-1) * ndim + n2;
                    index1 = index1(Sind(:,n1)); index2 = index2(Sind(:,n2));
                    swx2(n1,n2) = sum(sum(hmm.state(k).W.S_W(index1,index2) .* XXGXX{k}(Sind(:,n1),Sind(:,n2))));
                    swx2(n2,n1) = swx2(n1,n2);
                end
            end
        end
        hmm.Omega.Gam_rate(regressed,regressed) = hmm.Omega.Gam_rate(regressed,regressed) + (e + swx2(regressed,regressed));
    end
    hmm.Omega.Gam_irate(regressed,regressed) = inv(hmm.Omega.Gam_rate(regressed,regressed));
    hmm.Omega.Gam_shape = hmm.prior.Omega.Gam_shape + Tres;
    
else % state dependent
    for k=1:K
        setstateoptions;
        XWk = permute(XW(k,:,:),[2 3 1]);
        if train.uniqueAR
            e = (residuals - XWk).^2;
            swx2 = zeros(Tres,ndim);
            for n=1:ndim
                ind = n:ndim:size(XX{kk},2);
                swx2(:,n) = sum(XX{kk}(:,ind) * hmm.state(k).W.S_W .* XX{kk}(:,ind),2);
            end;
            hmm.state(k).Omega.Gam_rate = hmm.state(k).prior.Omega.Gam_rate + ...
                sum( repmat(Gamma(:,k),1,ndim) .* (e + swx2) ) / 2;
            hmm.state(k).Omega.Gam_shape = hmm.state(k).prior.Omega.Gam_shape + Gammasum(k) / 2;
                
        elseif strcmp(train.covtype,'diag')
            e = (residuals(:,regressed) - XWk(:,regressed)).^2;
            swx2 = zeros(Tres,ndim);
            if ~isempty(hmm.state(k).W.Mu_W)  
                for n=1:ndim
                    if ~regressed(n), continue; end
                    if ndim==1
                        swx2(:,n) = sum(XX{kk}(:,Sind(:,n)) * hmm.state(k).W.S_W(Sind(:,n),Sind(:,n)) ...
                            .* XX{kk}(:,Sind(:,n)),2);
                    else
                    swx2(:,n) = sum(XX{kk}(:,Sind(:,n)) * permute(hmm.state(k).W.S_W(n,Sind(:,n),Sind(:,n)),[2 3 1]) ...
                        .* XX{kk}(:,Sind(:,n)),2);
                    end
                end;
            end
            hmm.state(k).Omega.Gam_rate(regressed) = hmm.state(k).prior.Omega.Gam_rate(regressed) + ...
                sum( repmat(Gamma(:,k),1,sum(regressed)) .* (e + swx2(:,regressed) ) ) / 2;
            hmm.state(k).Omega.Gam_shape = hmm.state(k).prior.Omega.Gam_shape + Gammasum(k) / 2;
            
        else % full
            e = (residuals(:,regressed) - XWk(:,regressed));
            e = (e' .* repmat(Gamma(:,k)',sum(regressed),1)) * e;
            swx2 =  zeros(ndim,ndim);
            if ~isempty(hmm.state(k).W.Mu_W)  
                orders = formorders(train.order,train.orderoffset,train.timelag,train.exptimelag);
                if isempty(orders)
                    swx2 = hmm.state(k).W.S_W * XXGXX{k};
                else
                    for n1=find(regressed)
                        for n2=find(regressed)
                            if n2<n1, continue, end;
                            index1 = (0:length(orders)*ndim+(~train.zeromean)-1) * ndim + n1;
                            index2 = (0:length(orders)*ndim+(~train.zeromean)-1) * ndim + n2;
                            index1 = index1(Sind(:,n1)); index2 = index2(Sind(:,n2));
                            swx2(n1,n2) = sum(sum(hmm.state(k).W.S_W(index1,index2) .* XXGXX{k}(Sind(:,n1),Sind(:,n2))));
                            swx2(n2,n1) = swx2(n1,n2);
                        end
                    end
                end
            end
            hmm.state(k).Omega.Gam_rate(regressed,regressed) = hmm.state(k).prior.Omega.Gam_rate(regressed,regressed) + ...
                (e + swx2(regressed,regressed));
            hmm.state(k).Omega.Gam_irate(regressed,regressed) = inv(hmm.state(k).Omega.Gam_rate(regressed,regressed));
            hmm.state(k).Omega.Gam_shape = hmm.state(k).prior.Omega.Gam_shape + Gammasum(k);
        end
    end
    
end

end