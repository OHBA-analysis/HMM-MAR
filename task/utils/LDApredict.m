function [Y_pred, loglikelihoodX] = LDApredict(model,Gamma,X)
% For an inferred LDA model and state timecourses, computes the
% likelihood manifold in X space and computes the likelihood function for
% each class of labels.
[T2,K]=size(Gamma);
[T,nDim] = size(X);
for k=1:K
    betas_mu{k}=model.state(k).W.Mu_W((nDim+1):end,1:nDim);
    betas_sigma{k}=model.state(k).W.S_W(1:nDim,(nDim+1):end,(nDim+1):end); 
end

% Iterate through labels, noting that the distribution in x space can be
% computed as a sum for each condition:
Ydim = size(betas_mu{1},1)-1;
for testcond = 1:Ydim
    % compute each state's distribution in x space
    for k=1:K
        betamu_givenY(testcond,k,:) = betas_mu{k}(1,:)+betas_mu{k}(1+testcond,:);
        betasig_givenY(testcond,k,:,:) = diag(betas_sigma{k}(:,testcond+1,testcond+1)) + ...
            diag(betas_sigma{k}(:,1,1))+2*diag(betas_sigma{k}(:,1,testcond+1));
    end
end

% iterate through time, computing likelihood of X given Gamma:
if isfield(model,'Omega')
    if isfield(model.Omega,'pseudomean')
        Omega = model.Omega.pseudomean(1:nDim,1:nDim);
    else
        Omega = diag(model.Omega.Gam_shape ./ model.Omega.Gam_rate(1:nDim));
    end
else
    %insert code for statewise Omega estimation - still to do later
end
if T==T2
    for t=1:T
        if ~isfield(model,'Omega')
            for k=1:K
                Omega(:,:,k) = Gamma(t,k)*diag(model.state(k).Omega.Gam_shape ./ model.state(k).Omega.Gam_rate(1:nDim));
            end
            Omega=sum(Omega,3);
        end
        for testcond = 1:Ydim
            mu = sum(squeeze(betamu_givenY(testcond,:,:)) .*repmat(Gamma(t,:)',1,nDim));
            mu_rec(testcond,t,:)=mu;
            sig = Omega + squeeze(sum(squeeze(betasig_givenY(testcond,:,:,:)).*repmat(Gamma(t,:)',1,nDim,nDim),1));
            loglikelihoodX(t,testcond)=-0.5*log(det(sig)) -0.5 * (X(t,:)-mu)*inv(sig)*(X(t,:)-mu)';
        end
        m=max(loglikelihoodX(t,:),[],2);
        Y_pred(t,:) = loglikelihoodX(t,:)==m;
        if sum(Y_pred(t,:))>1
            fprintf(['\n\nWARNING: Equal scores achieved for multiple classes, t=',int2str(t),'\n\n']);
            a=find(Y_pred(t,:));
            Y_pred(t,a)=0;Y_pred(t,a(randi(length(a))))=1; %randomly silence all but one of these entries 
        end
    end
else
%implies Gamma invariant, so collapse over time:
    for testcond = 1:Ydim
        mu= squeeze(betamu_givenY(testcond,:,:))';
        mu_rec(testcond,:)=mu;
        sig = Omega + squeeze(betasig_givenY(testcond,:,:,:));
        loglikelihoodX(:,testcond)=-0.5*log(det(sig)) -0.5 * sum(((X-repmat(mu,T,1))*inv(sig)).*(X-repmat(mu,T,1)),2);
    end
    m=max(loglikelihoodX(:,:),[],2);
    Y_pred = loglikelihoodX(:,:)==repmat(m,1,Ydim);
end
end