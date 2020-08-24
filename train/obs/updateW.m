function [hmm,XW] = updateW(hmm,Gamma,residuals,XX,XXGXX,Tfactor) 

K = length(hmm.state); ndim = hmm.train.ndim;
if ~isempty(hmm.state(1).W.Mu_W)
    XW = zeros(size(XX,1),ndim,K);
else
    XW = [];
end
if nargin<6, Tfactor = 1; end
reweight = 0; % compensate for classes that have fewer instances?  
if isfield(hmm.train,'B'), Q = size(hmm.train.B,2);
else, Q = ndim; 
end
pcapred = hmm.train.pcapred>0;
if pcapred, M = hmm.train.pcapred; end
p = hmm.train.lowrank; do_HMM_pca = (p > 0);
setstateoptions;

if reweight % assumes there are two classes, encoded by -1 and 1   
    count = zeros(2,1); 
    count(1) = mean(residuals(:,end)<0);
    count(2) = mean(residuals(:,end)>0);
end

for k = 1:K
    
    if ~hmm.train.active(k), continue; end
    if isempty(orders) && train.zeromean && ~do_HMM_pca, continue; end
    if strcmp(train.covtype,'diag') || strcmp(train.covtype,'full'), omega = hmm.state(k).Omega;
    elseif ~isfield(train,'distribution') || ~strcmp(train.distribution,'logistic'), omega = hmm.Omega;
    end
    
    if reweight
        count_k = zeros(2,1);
        count_k(1) = sum( Gamma(:,k) .* (residuals(:,end)<0) ) / sum(Gamma(:,k));
        count_k(2) = sum( Gamma(:,k) .* (residuals(:,end)>0) ) / sum(Gamma(:,k));
        ratio = count ./ count_k;
        Gamma(residuals(:,end)<0,k) = Gamma(residuals(:,end)<0,k) * ratio(1);
        Gamma(residuals(:,end)>0,k) = Gamma(residuals(:,end)>0,k) * ratio(2);
    end
  
    if train.uniqueAR || ndim==1 % it is assumed that order>0 and cov matrix is diagonal
        if hmm.train.pcapred > 0, npred = hmm.train.pcapred;
        else npred = length(orders);
        end
        XY = zeros(npred+(~train.zeromean),1);
        XGX = zeros(npred+(~train.zeromean));
        for n = 1:ndim
            ind = n:ndim:size(XX,2);
            iomegan = omega.Gam_shape / omega.Gam_rate(n);
            XGX = XGX + iomegan * XXGXX{k}(ind,ind);
            XY = XY + (iomegan * XX(:,ind)' .* repmat(Gamma(:,k)',length(ind),1)) * residuals(:,n);
        end
        if ~isempty(train.prior)
            hmm.state(k).W.S_W = inv(train.prior.iS + XGX);
            hmm.state(k).W.Mu_W = hmm.state(k).W.S_W * (XY + train.prior.iSMu); % order by 1
        else
            if train.zeromean==0 && pcapred
                regterm = diag([hmm.state(k).prior.Mean.iS; (hmm.state(k).beta.Gam_shape ./ ...
                    hmm.state(k).beta.Gam_rate) ]);  
            elseif pcapred
                regterm = diag((hmm.state(k).beta.Gam_shape ./  hmm.state(k).beta.Gam_rate));
            elseif train.zeromean==0 && ~isempty(orders)
                regterm = diag([hmm.state(k).prior.Mean.iS (hmm.state(k).alpha.Gam_shape ./ ...
                    hmm.state(k).alpha.Gam_rate) ]);
            elseif train.zeromean==0
                regterm = diag(hmm.state(k).prior.Mean.iS);
            else
                regterm = diag((hmm.state(k).alpha.Gam_shape ./  hmm.state(k).alpha.Gam_rate));
            end
            hmm.state(k).W.S_W = inv(regterm + Tfactor * XGX);
            hmm.state(k).W.Mu_W = Tfactor * hmm.state(k).W.S_W * XY; % order by 1
        end        
        for n = 1:ndim
            ind = n:ndim:size(XX,2);
            XW(:,n,k) = XX(:,ind) * hmm.state(k).W.Mu_W;
        end
                
    elseif strcmp(train.covtype,'diag') || strcmp(train.covtype,'uniquediag')
        for n = 1:ndim
            ndim_n = sum(S(:,n)>0);
            if ndim_n==0 && train.zeromean==1, continue; end
            regterm = [];
            if ~train.zeromean, regterm = hmm.state(k).prior.Mean.iS(n); end
            if ~isempty(orders)
                if pcapred
                    regterm = [regterm; hmm.state(k).beta.Gam_shape ./ hmm.state(k).beta.Gam_rate(:,n)];
                else
                    alphaterm = repmat( (hmm.state(k).alpha.Gam_shape ./  hmm.state(k).alpha.Gam_rate), ndim_n, 1);
                    if ndim>1
                        regterm = [regterm; repmat(hmm.state(k).sigma.Gam_shape(S(:,n),n) ./ ...
                            hmm.state(k).sigma.Gam_rate(S(:,n),n), length(orders), 1).*alphaterm(:) ];
                    else
                        regterm = [regterm; alphaterm(:)];
                    end
                end
            end
            if isempty(regterm), regterm = 0; end
            regterm = diag(regterm);
            hmm.state(k).W.iS_W(n,Sind(:,n),Sind(:,n)) = ...
                regterm + Tfactor * (omega.Gam_shape / omega.Gam_rate(n)) * XXGXX{k}(Sind(:,n),Sind(:,n));
            hmm.state(k).W.iS_W(n,Sind(:,n),Sind(:,n)) = (permute(hmm.state(k).W.iS_W(n,Sind(:,n),Sind(:,n)),[2 3 1]) + ...
                permute(hmm.state(k).W.iS_W(n,Sind(:,n),Sind(:,n)),[2 3 1])' ) / 2; % ensuring symmetry
            hmm.state(k).W.S_W(n,Sind(:,n),Sind(:,n)) = ...
                inv(permute(hmm.state(k).W.iS_W(n,Sind(:,n),Sind(:,n)),[2 3 1]));
            hmm.state(k).W.Mu_W(Sind(:,n),n) = (( permute(hmm.state(k).W.S_W(n,Sind(:,n),Sind(:,n)),[2 3 1]) * ...
                Tfactor * (omega.Gam_shape / omega.Gam_rate(n)) * XX(:,Sind(:,n))') .* ...
                repmat(Gamma(:,k)',sum(Sind(:,n)),1)) * residuals(:,n);
        end
        XW(:,:,k) = XX * hmm.state(k).W.Mu_W;
        
    elseif isfield(train,'distribution') && strcmp(train.distribution,'logistic')
        
        % Set Y and X: 
        Xdim = size(XX,2)-hmm.train.logisticYdim;
        X = XX(:,1:Xdim);
        Y = residuals;
        vp = Y~=0; % for multinomial logistic regression, only include valid points
        if hmm.train.balancedata
            w=(1/(hmm.train.origlogisticYdim))*sum(Gamma(:,k))./(sum([Y==1] .* Gamma(:,k)));%(1+hmm.train.origlogisticYdim));
            w_star=((hmm.train.origlogisticYdim-1)/hmm.train.origlogisticYdim)*sum(Gamma(:,k))./(sum([Y==-1] .* Gamma(:,k)));
            weightvector = [Y==1].*w + [Y==-1].*w_star;
            Gammaweighted=Gamma(vp,k) .*weightvector;
        else
            Gammaweighted=Gamma(vp,k);
        end
        % initialise priors - with ARD:
        W_mu0 = zeros(Xdim,1);
        W_sig0 = diag(hmm.state(k).alpha.Gam_shape ./ hmm.state(k).alpha.Gam_rate(1:Xdim));
        
        % implement update equations for logistic regression:
        lambdafunc = @(psi_t) ((2*psi_t).^-1).*(log_sigmoid(psi_t)-0.5);
        
        %select functioning channels:
        for n = 1:ndim
            ndim_n = sum(S(:,n));
            if ndim_n==0, continue; end
            WW=cell(K,1);
            for i=1:K
                WW{i}=hmm.state(i).W.Mu_W(Sind(:,n),n)*hmm.state(i).W.Mu_W(Sind(:,n),n)' + ...
                            squeeze(hmm.state(i).W.S_W(n,S(:,n),S(:,n)));
            end
            if ~isfield(hmm,'psi')
                hmm = updatePsi(hmm,Gamma,X,Y);
            end
            % note this could be optimised with better use of XXGXX:
%             W_sigsum{k}=zeros(T,ndim_n,ndim_n);
%             for t=1:T
%                 W_sigsum{k}(t,:,:)=2*lambdafunc(hmm.psi(t))*Gamma(t,k)*X(t,:)'*X(t,:);
%             end
            W_sigsum = (XX(vp,1:ndim_n)' .* repmat(2*lambdafunc(hmm.psi(vp))'.*Gammaweighted',ndim_n,1))* XX(vp,1:ndim_n);
            %update parameter entries:
            hmm.state(k).W.S_W(n,S(:,n),S(:,n)) = inv(squeeze(W_sigsum)+inv(W_sig0));
            hmm.state(k).W.Mu_W(S(:,n),n) = squeeze(hmm.state(k).W.S_W(n,S(:,n),S(:,n))) * 0.5 * X(vp,:)' * (Y(vp).*Gammaweighted) ... %sum(W_musum{k},1)') ...
                +(W_sig0\W_mu0);
            
            % Also increment optimal tuning parameters psi:
             WWupdate = hmm.state(k).W.Mu_W(Sind(:,n),n)*hmm.state(k).W.Mu_W(Sind(:,n),n)' + ...
                              squeeze(hmm.state(k).W.S_W(n,S(:,n),S(:,n))) - WW{k};
             psiupdate = sum(((X .* repmat(Gamma(:,k),1,size(X,2))) * WWupdate).*X , 2);
             hmm.psi = sqrt(hmm.psi.^2+psiupdate);
        end
        
    else % full or unique full
        if all(S(:)==1)
            if pcapred
                mlW = (( XXGXX{k} \ XX') .* repmat(Gamma(:,k)',(~train.zeromean)+M,1) * residuals)';
            else
                mlW = (( XXGXX{k} \ XX') .* repmat(Gamma(:,k)',...
                    (~train.zeromean)+Q*length(orders),1) * residuals)';
            end
            regterm = [];
            if ~train.zeromean, regterm = hmm.state(k).prior.Mean.iS; end % ndim by 1
            if ~isempty(orders) 
                if pcapred
                    betaterm = (hmm.state(k).beta.Gam_shape ./ hmm.state(k).beta.Gam_rate)';
                    regterm = [regterm; betaterm(:)];
                else
                    sigmaterm = (hmm.state(k).sigma.Gam_shape ./ hmm.state(k).sigma.Gam_rate)'; 
                    sigmaterm = sigmaterm(:); 
                    sigmaterm = repmat(sigmaterm, length(orders), 1); % ndim*ndim*order by 1 
                    alphaterm = repmat( (hmm.state(k).alpha.Gam_shape ./ hmm.state(k).alpha.Gam_rate), ...
                        length(hmm.state(k).sigma.Gam_rate(:)), 1);
                    alphaterm = alphaterm(:);
                    regterm = [regterm; (alphaterm .* sigmaterm)];
                end
            end
            if isempty(regterm), regterm = 0; end
            regterm = diag(regterm);
            prec = omega.Gam_shape * omega.Gam_irate;
            gram = kron(XXGXX{k}, prec);
            hmm.state(k).W.iS_W = regterm + Tfactor * gram;
            hmm.state(k).W.S_W = (hmm.state(k).W.S_W + hmm.state(k).W.S_W') / 2; 
            hmm.state(k).W.S_W = inv(hmm.state(k).W.iS_W);
            muW = Tfactor * hmm.state(k).W.S_W * gram * mlW(:);
            if pcapred
                hmm.state(k).W.Mu_W = reshape(muW,ndim,(~train.zeromean)+M)';
            else
                hmm.state(k).W.Mu_W = reshape(muW,ndim,~train.zeromean+Q*length(orders))';
            end
            XW(:,:,k) = XX * hmm.state(k).W.Mu_W;
        else
            
            dependentvariables = sum(S,1)>0;
            independentvariables = sum(S,2)>0;
            Ydim = sum(any(S,1));
            Xdim = sum(any(S,2));
            Y = residuals(:,dependentvariables);
            X = XX(:,independentvariables);
            
            prec = omega.Gam_shape * omega.Gam_irate(dependentvariables,dependentvariables);
            % note that XXGXX is invalid if any S==0:
            temp = (bsxfun(@times,X,Gamma(:,k)))' * X;
            gram = kron(prec,temp);
            
            % Regularisation type:
            if strcmp(hmm.train.regularisation,'ARD')
                sigmaterm = (hmm.state(k).sigma.Gam_shape(S) ./ hmm.state(k).sigma.Gam_rate(S))'; %ARD prior over M values
                sigmaterm = sigmaterm(:);
                alphaterm = repmat( (hmm.state(k).alpha.Gam_shape ./ hmm.state(k).alpha.Gam_rate), ...
                    length(sigmaterm),1);
                regterm = diag(alphaterm .* sigmaterm);
            elseif strcmp(hmm.train.regularisation,'Ridge')
                sigmaterm = ones(Ydim,Xdim);
                sigmaterm = diag(sigmaterm(:)); 
                regterm = sigmaterm;
            elseif strcmp(hmm.train.regularisation,'Sparse')
                error('Sparse L1 regularisation not yet implemented');
            end
            
            hmm.state(k).W.iS_W = zeros(length(S(:)));
            hmm.state(k).W.S_W = zeros(length(S(:)));
            validentries = logical(S(:));
            hmm.state(k).W.iS_W(validentries,validentries) = regterm + gram;
            hmm.state(k).W.S_W(validentries,validentries) = inv(hmm.state(k).W.iS_W(validentries,validentries));
            hmm.state(k).W.iS_W = sparse(hmm.state(k).W.iS_W);
            hmm.state(k).W.S_W = sparse(hmm.state(k).W.S_W);
            
            % and compute mean:
            temp = (bsxfun(@times,X,Gamma(:,k)))' * Y * prec;
            muW = hmm.state(k).W.S_W(validentries,validentries)*squash(temp);
            muW = reshape(muW,Xdim,Ydim);
            
            hmm.state(k).W.Mu_W = zeros(size(S));
            hmm.state(k).W.Mu_W(S) = muW;
            
            XW(:,:,k) = XX * hmm.state(k).W.Mu_W;
        end
    end
    
end

end
