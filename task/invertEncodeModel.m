function [Y_pred, loglikelihoodX] = invertEncodeModel(model,Gamma,X)
% For an inferred encoding model and state timecourses, computes the
% likelihood manifold in X space and computes the liklihood function for
% each class of labels.
[T,K]=size(Gamma);
nDim = size(X,2);
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
%         betasig_givenY(testcond,k,:,:) = diag(diag(squeeze(betas_sigma{k}(testcond,2:end,2:end)))) + ...
%             diag(betas_sigma{k}(testcond,1,1)+2*squeeze(betas_sigma{k}(testcond,1,2:end)));
        betasig_givenY(testcond,k,:,:) = diag(betas_sigma{k}(:,testcond+1,testcond+1)) + ...
            diag(betas_sigma{k}(:,1,1))+2*diag(betas_sigma{k}(:,1,testcond+1));
    end
end

% iterate through time, computing likelihood of X given Gamma:
Omega = diag(model.Omega.Gam_shape ./ model.Omega.Gam_rate(1:nDim));
for t=1:T
    for testcond = 1:Ydim
        mu = sum(squeeze(betamu_givenY(testcond,:,:)) .*repmat(Gamma(t,:)',1,nDim));
        mu_rec(testcond,t,:)=mu;
        sig = Omega + squeeze(sum(squeeze(betasig_givenY(testcond,:,:,:)).*repmat(Gamma(t,:)',1,nDim,nDim),1));
        loglikelihoodX(t,testcond)=-0.5*log(det(sig)) -0.5 * (X(t,:)-mu)*inv(sig)*(X(t,:)-mu)';
    end
    m=max(loglikelihoodX(t,:),[],2);
    Y_pred(t,:) = loglikelihoodX(t,:)==m;
    if sum(Y_pred(t,:))>1
        fprintf('\n\nWARNING: Equal scores achieved for multiple classes\n\n');
        a=find(Y_pred(t,:));
        Y_pred(t,a)=0;Y_pred(t,a(randi(length(a))))=1; %randomly silence all but one of these entries 
        
    end
end


end