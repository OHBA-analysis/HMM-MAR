function [Y_pred, loglikelihoodX] = LDApredict(model,Gamma,X,classification,intercept)
% For an inferred LDA model and state timecourses, computes the
% likelihood manifold in X space and computes the likelihood function for
% each class of labels.
if nargin<4
    classification=true;
end
if nargin<5
    intercept=true;
end
[T2,K] = size(Gamma);
[T,nDimX] = size(X);
nDimY = size(model.state(1).W.Mu_W,2)-nDimX;
betas_mu = cell(K,1);
betas_sigma = cell(K,1);
for k = 1:K
    betas_mu{k} = model.state(k).W.Mu_W((nDimX+1):end,1:nDimX);
    if strcmp(model.train.covtype,'full') || strcmp(model.train.covtype,'uniquefull')
        S = model.train.S==1;
        betas_sigma{k} = model.state(k).W.S_W(logical(S(:)),logical(S(:)));
    else
        betas_sigma{k} = model.state(k).W.S_W(1:nDimX,(nDimX+1):end,(nDimX+1):end); 
    end
end

% Iterate through labels, noting that the distribution in x space can be
% computed as a sum for each condition:
if intercept
    numconds = size(betas_mu{1},1)-1;
else
    numconds = size(betas_mu{1},1);
end
betamu_givenY = zeros(numconds,K,nDimX);
betasig_givenY = zeros(numconds,K,nDimX,nDimX);
for testcond = 1:numconds
    % compute each state's distribution in x space - assuming intercept has
    % been used
    for k=1:K
        if classification && intercept %include mean in generative model:
            betamu_givenY(testcond,k,:) = betas_mu{k}(1,:) + betas_mu{k}(1 + testcond,:);
        elseif intercept
            betamu_givenY(testcond,k,:) = betas_mu{k}(1 + testcond,:);
        else
            betamu_givenY(testcond,k,:) = betas_mu{k}(testcond,:);
        end
        if strcmp(model.train.covtype,'full') || strcmp(model.train.covtype,'uniquefull')
            betasig_givenY(testcond,k,:,:) = betas_sigma{k}([1:nDimY:nDimY*nDimX],[1:nDimY:nDimY*nDimX])+...
                betas_sigma{k}([testcond:nDimY:nDimY*nDimX],[testcond:nDimY:nDimY*nDimX]);
        else % Naive bayes classifier - just take uncertainty on each channel
            betasig_givenY(testcond,k,:,:) = diag(betas_sigma{k}(:,testcond + 1,testcond + 1)) + ...
                diag(betas_sigma{k}(:,1,1)) + 2*diag(betas_sigma{k}(:,1,testcond + 1));
        end
    end
end

% iterate through time, computing likelihood of X given Gamma:
if isfield(model,'Omega')
    if isfield(model.Omega,'pseudomean')
        CovMat = model.Omega.pseudomean(1:nDimX,1:nDimX);
    elseif strcmp(model.train.covtype,'uniquediag')  % Naive Bayes case
        prec = diag(model.Omega.Gam_shape ./ model.Omega.Gam_rate(1:nDimX));
        CovMat = inv(prec);
    else % unique full covariance matrix
        prec = model.Omega.Gam_shape * model.Omega.Gam_irate(1:nDimX,1:nDimX);
        CovMat = inv(prec);
    end
else
    if strcmp(model.train.covtype,'diag')
        for k=1:K
            prec = diag(model.state(k).Omega.Gam_shape ./ model.state(k).Omega.Gam_rate(1:nDimX));
            CovMat(:,:,k) = inv(prec);
        end
    else % statewise full covariance
        for k=1:K
            prec = model.state(k).Omega.Gam_shape * model.state(k).Omega.Gam_irate(1:nDimX,1:nDimX);
            CovMat(:,:,k) = inv(prec);
        end
    end
end

% remove intercept from data if necessary:
if intercept
    for k=1:K
        X = X - Gamma(:,k) * betas_mu{k}(1,:);
    end
    Y_pred = zeros(T,nDimY-1);
else
    Y_pred = zeros(T,nDimY);
end

if T==T2
    for t=1:T
        CovMat_t = sum(CovMat .* repmat(permute(Gamma(t,:),[3,1,2]),nDimX,nDimX,1),3);
        if classification
            for testcond = 1:numconds
                mu = sum(squeeze(betamu_givenY(testcond,:,:)) .*repmat(Gamma(t,:)',1,nDimX));
                mu_rec(testcond,t,:) = mu;
                S = CovMat_t + squeeze(sum(squeeze(betasig_givenY(testcond,:,:,:)).*repmat(Gamma(t,:)',1,nDimX,nDimX),1));
                loglikelihoodX(t,testcond) = -0.5*log(det(S)) -0.5 * (X(t,:) - mu) * inv(S) * (X(t,:) - mu)';
            end
            m = max(loglikelihoodX(t,:),[],2);
            Y_pred(t,:) = loglikelihoodX(t,:)==m;
            if sum(Y_pred(t,:))>1
                warning(['Equal scores achieved for multiple classes, t=',int2str(t),'\n\n']);
                a = find(Y_pred(t,:));
                Y_pred(t,a) = 0;
                Y_pred(t,a(randi(length(a)))) = 1; %randomly silence all but one of these entries 
            end
        else
            betas_t = permute(sum(repmat(Gamma(t,:),[numconds,1,nDimX]).* betamu_givenY,2),[1,3,2]);
%             Y_pred(t,:) = (betas_t * inv(CovMat_t) * X(t,:)') ...
%                *inv(betas_t * inv(CovMat_t) * betas_t');
            
            %alternative implementation:
            if ~intercept
                sigma_Y = inv(inv(eye(nDimY)) + betas_t * inv(CovMat_t) * betas_t');
            else
                sigma_Y = inv(inv(eye(nDimY-1)) + betas_t * inv(CovMat_t) * betas_t');
            end
            betas_backwardmodel = inv(CovMat_t) * betas_t' * sigma_Y;
            Y_pred(t,:) = X(t,:)*betas_backwardmodel;
        end
    end
else
%implies Gamma invariant, so collapse over time:
    for testcond = 1:numconds
        mu = squeeze(betamu_givenY(testcond,:,:))';
        mu_rec(testcond,:) = mu;
        sig = CovMat + squeeze(betasig_givenY(testcond,:,:,:));
        loglikelihoodX(:,testcond) = -0.5 * log(det(sig)) -0.5 * sum(((X-repmat(mu,T,1))*inv(sig)).*(X-repmat(mu,T,1)),2);
    end
    m = max(loglikelihoodX(:,:),[],2);
    Y_pred = loglikelihoodX(:,:)==repmat(m,1,numconds);
end
if numconds==2 && all(Y_pred(:,1) == 1-Y_pred(:,2))
    % binary output:
    Y_pred = Y_pred(:,1);
end
end