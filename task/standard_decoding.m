function [cv_acc,acc,genplot,model] = standard_decoding(X,Y,T,options,binsize)
% Compute cross-validated (CV) accuracies from a "decoding" model trained 
% at each time point (or window, see below) in the trial  
% The reported statistic is either the cross-validated explained variance from
% regressing X on the stimulus (default), or the Pearson correlation between the
% predictions and the true values (if options.accuracyType='Pearson');
%
% INPUT
% X: Brain data, (total time by regions) or (time by trials by regions)
% Y: Stimulus, (total time by q), where q is no. of stimulus features; OR
%              (no.trials by q), meaning that each trial has a single
%              stimulus value for the entire length of the trial
% T: Length of series
% options: structure with the preprocessing options - see documentation in 
%                       https://github.com/OHBA-analysis/HMM-MAR/wiki
% binsize: how many consecutive time points will be used for each estiamtion. 
%           By default, this is 1, such that one decoding model
%               is estimated using 1 time point
%
% OUTPUT
% cv_acc:  (time points by stimuli) time series of CV- explained variance 
%           or classification accuracies
% acc:     (time points by stimuli) time series of non-CV explained variance
%           or classification accuracies
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2018)

if nargin < 4 || isempty(options), options = struct(); end
if nargin < 5 || isempty(binsize), binsize = 1; end

if ~all(T==T(1)), error('All elements of T must be equal for cross validation'); end 
N = length(T); ttrial = T(1); 

if size(Y,1) < size(Y,2); Y = Y'; end
q = size(Y,2);
if size(Y,1) == length(T) % one value per trial
    responses = Y;
    Ystar = zeros(ttrial,N,q);
    for t = 1:ttrial, Ystar(t,:,:) = Y; end
    Ystar = reshape(Ystar,[ttrial*N q]);
elseif length(Y(:)) ~= (ttrial*N*q)
    error('Incorrect dimensions in Y')
else
    responses = reshape(Y,[ttrial N q]);
    Ystar = Y; % save for later addition of intercept term
    responses = permute(responses(1,:,:),[2 3 1]); % N x q
    if ~all(Y(:)==Ystar(:))
%        error('For cross-validating, the same stimulus must be presented for the entire trial');
    end
end

max_num_classes = 5;
classification = length(unique(responses(:))) < max_num_classes ;
if ~isfield(options,'temporalgeneralisation')
    options.temporalgeneralisation = false; 
end

if classification
    Ycopy = Y;
    if size(Ycopy,1) == N 
        Ycopy = repmat(reshape(Ycopy,[1 N q]),[ttrial 1 1]);
    else
        Ycopy = reshape(Ycopy,[ttrial N q]);
    end
    % no demeaning by default if this is a classification problem
    if ~isfield(options,'demeanstim'), options.demeanstim = 0; end
end

options.Nfeatures = 0; 
options.K = 1; 
[X,Y,T] = preproc4hmm(X,Y,T,options); % this demeans Y
p = size(X,2);
qstar = size(Y,2); % qstar = q if no intercept term; qstar = q+1 if intercept used

X = reshape(X,[ttrial N p]);
Y = reshape(Y,[ttrial N qstar]);

if mod(binsize/2,2)==0
    warning(['binsize must be an odd number, setting to ' num2str(binsize+1)])
    binsize = binsize + 1; 
end

if any(T~=ttrial), error('All trials must have the same length'); end
if nargin<5, binsize = 1; end

if isfield(options,'NCV')
    NCV = options.NCV; options = rmfield(options,'NCV');
else, NCV = 10; 
end
if isfield(options,'lambda') 
    lambda = options.lambda; options = rmfield(options,'lambda');
else, lambda = 0.0001; 
end
if isfield(options,'lossfunc') 
    lossfunc = options.lossfunc; options = rmfield(options,'lossfunc');
else, lossfunc = 'quadratic'; 
end
% option to first fit an encoding model then decode - ie Linear Guassian System approach
if ~isfield(options,'encodemodel'),options.encodemodel = false;end
if ~isfield(options,'SVMreg'),options.SVMreg = false;
elseif (options.SVMreg & ~isfield(options,'SVMkernel')),options.SVMkernel = 'linear',end
    
% Form CV folds; if response are categorical, then it's stratified
if ~isfield(options,'c')
    if classification
        %disp('Response is treated as categorical')
        tmp = zeros(N,1);
        for j = 1:q
            rj = responses(:,j);
            uj = unique(rj);
            for jj = 1:length(uj)
                tmp(rj == uj(jj)) = tmp(rj == uj(jj)) + (q+1)^(j-1) * jj; 
                %tmp(rj == uj(jj)) + 100^(j-1) * jj;
            end
        end
        uy = unique(tmp);
        group = zeros(N,1);
        for j = 1:length(uy)
            group(tmp == uy(j)) = j;
        end
        c2 = cvpartition(group,'KFold',NCV);
    else
        c2 = cvpartition(N,'KFold',NCV);
        %disp('Response is treated as continuous - no CV stratification')
    end
    c = struct();
    c.test = cell(NCV,1);
    c.training = cell(NCV,1);
    for icv = 1:NCV
        c.training{icv} = c2.training(icv);
        c.test{icv} = c2.test(icv);
    end; clear c2
else
    c = options.c;  
    NCV = length(c.test);
end

halfbin = floor(binsize/2); 
%nwin = round(ttrial / binsize);
%binsize = floor(ttrial / nwin); 
if options.encodemodel
    RidgePen = lambda * eye(qstar);
    Ysig = eye(qstar); % prior over design matrix magnitude - this can also be determined for each cv fold
else
    RidgePen = lambda * eye(p);
end

% Perform the prediction 
model = [];
if ~options.encodemodel
    Ypred = NaN(ttrial,N,qstar);
else
    Ypred = NaN(ttrial,N,q);
    model.beta_encode = zeros(qstar,p,ttrial);
    model.noise_encode = zeros(p,p,ttrial);
end
model.beta_decode = zeros(q,p,ttrial);
beta = cell(length(halfbin+1 : ttrial-halfbin),NCV);
for icv = 1:NCV
    Ntr = sum(c.training{icv}); Nte = sum(c.test{icv});
    for t = halfbin+1 : ttrial-halfbin
        r = t-halfbin:t+halfbin;
        %r = (1:binsize) + (t-1)*binsize;
        Xtr = reshape(X(r,c.training{icv},:),binsize*Ntr,p);
        Ytr = reshape(Y(r,c.training{icv},:),binsize*Ntr,qstar);
        Xte = reshape(X(t,c.test{icv},:),Nte,p);
        if options.encodemodel
            % fit encoding model, then convert to equivalent decode weights
            beta_encode = (Ytr' * Ytr + RidgePen) \ (Ytr' * Xtr);
            noise_X_t = cov(Xtr - Ytr*beta_encode) + 1e-5*eye(p);
            if qstar>q
                % regress out the (fixed) intercept value:
                Xte = Xte - ones(Nte,1) * beta_encode(1,:);
                sig_Y_t = inv(inv(Ysig(2:end,2:end)) + beta_encode(2:end,:) * inv(noise_X_t) * beta_encode(2:end,:)');
                beta{t,icv} = inv(noise_X_t) * beta_encode(2:end,:)' * sig_Y_t;
            else
                sig_Y_t = inv(inv(Ysig) + beta_encode * inv(noise_X_t) * beta_encode');
                beta{t,icv} = inv(noise_X_t) * beta_encode' * sig_Y_t;
            end
            model.beta_encode(:,:,t) = model.beta_encode(:,:,t) + beta_encode./NCV;
            model.noise_encode(:,:,t) = model.noise_encode(:,:,t) + noise_X_t./NCV;
        elseif options.SVMreg
            for i=1:q
                beta{t,icv,i} = fitrsvm(Xtr,Ytr(:,i),'KernelFunction',options.SVMkernel);
            end
        else
            beta{t,icv} = (Xtr' * Xtr + RidgePen) \ (Xtr' * Ytr);
        end
        if ~options.SVMreg
            Ypred(t,c.test{icv},:) = reshape(Xte * beta{t,icv},Nte,q);
            model.beta_decode(:,:,t) = squeeze(model.beta_decode(:,:,t)) + beta{t,icv}'./NCV;
        else
            for i=1:q
                Ypred(t,c.test{icv},i) = predict(beta{t,icv,i},Xte);
            end
        end
    end
end

% Compute CV accuracy / explained variance
cv_acc = NaN(ttrial,q);
Ystar = reshape(Ystar,[ttrial,N,q]);
for t = halfbin+1 : ttrial-halfbin
    Yt = reshape(Ystar(t,:,:),N,q);
    Ypredt = reshape(Ypred(t,:,:),N,q);
    
    if classification
        Ycopyt = reshape(Ycopy(t,:,:),N,q);
        Ypredt_star = continuous_prediction_2class(Ycopyt,Ypredt);
        if q == 1
            Ycopyt = 2*(Ycopyt>0)-1;
            cv_acc(t) = mean(abs(Ycopyt - Ypredt_star) < 1e-4);
        else
            cv_acc(t) = mean(sum(abs(Ycopyt - Ypredt_star),2) < 1e-4);
        end        
    else
        if q == 1
            if isfield(options,'accuracyType') && strcmp(options.accuracyType,'Pearson')
                cv_acc(t,:) = diag(corr(Yt,Ypredt));
            else
                cv_acc(t,:) = 1 - sum((Yt - Ypredt).^2) ./ sum(Yt.^2);
            end
        else
            if isfield(options,'accuracyType') && strcmp(options.accuracyType,'Pearson')
                cv_acc(t,:) = diag(corr(Yt,Ypredt));
            else
                cv_acc(t,:) = 1 - sum((Yt - Ypredt).^2) ./ sum(Yt.^2);
            end
        end
    end
end
% compute temporal generalisation plots
genplot = [];
if options.temporalgeneralisation
    Ypred = zeros(length(halfbin+1 : ttrial-halfbin),length(halfbin+1 : ttrial-halfbin),N,q);
    for icv = 1:NCV
        Ntr = sum(c.training{icv}); Nte = sum(c.test{icv});
        for t_train = halfbin+1 : ttrial-halfbin
            beta_temp = beta{t_train,icv};
            for t_test = halfbin+1 : ttrial-halfbin
                Xte = reshape(X(t_test,c.test{icv},:),Nte,p);
                Ypred(t_test,t_train,c.test{icv},:) = reshape(Xte * beta_temp,Nte,q);
            end
        end
    end
    
    for t_train = halfbin+1 : ttrial-halfbin
        Yt = reshape(Ystar(t_train,:,:),N,q);
        if classification; Ycopyt = reshape(Ycopy(t_train,:,:),N,q); end
        for t_test = halfbin+1 : ttrial-halfbin
            Ypredt_test = reshape(Ypred(t_test,t_train,:,:),N,q);
            if classification
                Ypredt_star = continuous_prediction_2class(Ycopyt,Ypredt_test);
                if q == 1
                    genplot(t_test,t_train) = mean(abs(Ycopyt - Ypredt_star) < 1e-4);
                else
                    genplot(t_test,t_train) = mean(sum(abs(Ycopyt - Ypredt_star),2) < 1e-4);
                end        
            else
                if isfield(options,'accuracyType') && strcmp(options.accuracyType,'Pearson')
                    genplot(t_test,t_train,:) = diag(corr(Yt,Ypredt_test));
                else
                    genplot(t_test,t_train,:) = 1 - sum((Yt - Ypredt_test).^2) ./ sum(Yt.^2);
                end
            end
        end
    end
end

% non-cross validated
acc = NaN(ttrial,q);
Ypred = zeros(size(Ystar));

for t = halfbin+1 : ttrial-halfbin
    r = t-halfbin:t+halfbin;
    Xt = reshape(X(r,:,:),binsize*N,p);
    Yt = reshape(Y(r,:,:),binsize*N,qstar);
    if options.encodemodel
        % fit encoding model, then convert to equivalent decode weights
        beta_encode = (Yt' * Yt + RidgePen) \ (Yt' * Xt);
        noise_X_t = cov(Xt - Yt*beta_encode) + 1e-5*eye(p);
        if qstar>q
            % regress out the (fixed) intercept value:
            Xt = Xt - ones(N,1) * beta_encode(1,:);
            sig_Y_t = inv(inv(Ysig(2:end,2:end)) + beta_encode(2:end,:) * inv(noise_X_t) * beta_encode(2:end,:)');
            beta_t = inv(noise_X_t) * beta_encode(2:end,:)' * sig_Y_t;
        else
            sig_Y_t = inv(inv(Ysig) + beta_encode * inv(noise_X_t) * beta_encode');
            beta_t = inv(noise_X_t) * beta_encode' * sig_Y_t;
        end
        Ypred(t,:,:) = reshape(Xt * beta_t,N,q);
    else
        beta_t = (Xt' * Xt + RidgePen) \ (Xt' * Yt);
        Xt = reshape(X(t,:,:),N,p);
        Ypred(t,:,:) = reshape(Xt * beta_t,N,q);
    end
   
    Yt = reshape(Ystar(t,:,:),N,q);
    Ypredt = permute(Ypred(t,:,:),[2 3 1]);
    if classification
        Ycopyt = reshape(Ycopy(t,:,:),N,q);
        Ypredt_star = continuous_prediction_2class(Ycopyt,Ypredt);
        if q == 1
            acc(t) = mean(abs(Ycopyt - Ypredt_star) < 1e-4);
        else
            acc(t) = mean(sum(abs(Ycopyt - Ypredt_star),2) < 1e-4);
        end
    else
        if q == 1
            if isfield(options,'accuracyType') && strcmp(options.accuracyType,'Pearson')
                acc(t) = diag(corr(Yt,Ypredt));
            else
                acc(t) = 1 - sum((Yt - Ypredt).^2) ./ sum(Yt.^2);
            end
        else
            if isfield(options,'accuracyType') && strcmp(options.accuracyType,'Pearson')
                acc(t,:) = diag(corr(Yt,Ypredt));
            else
                acc(t,:) = 1 - sum((Yt - Ypredt).^2) ./ sum(Yt.^2);
            end
        end
        
    end
end


end