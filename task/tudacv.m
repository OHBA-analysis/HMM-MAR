function [R2,pval,surrogates,c] = tudacv(X,Y,T,options,Gamma)
%
% Performs cross-validation of the TUDA model, which can be useful for
% example to compare different number or states or other parameters
% Cross-validation has a limitation for TUDA that is acknowledged in
%   Vidaurre et al (2017). ***TO COMPLETE***
%   (section 'Modelling between-trial temporal differences improves decoding
%   performance')
% NOTE: Specificying Nfeatures will lead to a circular assessment 
% (overfitting of R2), so this is discouraged
% 
% INPUT
% X: Brain data, (time by regions)
% Y: Stimulus, (time by q); q is no. of stimulus features
% T: Length of series
% options: structure with the training options - see documentation in 
%                       https://github.com/OHBA-analysis/HMM-MAR/wiki
%  Apart from the options specified for tudatrain, specific to tudacv are:
%  - options.mode, referring to how to do cross-validation. 
%                  if 1, CV is done time point by time point, across CV folds, 
%                   to produce a (time by 1) accuracy vector. 
%                  if 2, CV is performed across time points, so we would get
%                   a (time by time) accuracy "generalisation" matrix.
%  - options.conservative, whether or not to do conservative CV analysis.
%                  The issue is that the state time courses are estimated
%                  using both the data and the stimuli; then in the
%                  held-out data there is no way to know which state is
%                  active because, by definition, we do not use the
%                  stimulus (that is what we want to predict). 
%                  . if 1, then the state active at each time point is taken
%                  to be the one that is most active in training at each
%                  time point (losing between-trial variability)
%                  . if 0, then the actual state time courses are used. 
%  - options.estimation_method, if options.conservative == 1, this dictates how to
%                  estimate the state time courses in the left-out data:
%                  . if 1, then it uses the state active at each time point is taken
%                  to be the one that is most active in training at each
%                  time point (losing between-trial variability)
%                  . if 2, the state time courses are estimated using
%                  the just data and linear regression
%  - options.NCV, containing the number of cross-validation folds (default 10)
%  - options.lambda, regularisation penalty for the decoding
%  - options.lossfunc, loss function to compute the cross-validated error, 
%                  default is 'quadratic', in which case R2 corresponds to
%                  explained variance. Other options are: 'absolute', 'huber'
%  - options.Nperm, if options.mode==2, it generate surrogates by permuting
%                   the state time courses across trials, such that the
%                   average remains the same but the trial-specific
%                   temporal features are lost
% Gamma: Precomputed decoding models time courses, used if options.mode == 2 
%           (optional)
%
% OUTPUT 
% R2: cross-validated explained variance
% c: CV folds structure (c.training, c.test)
% pval: if options.Nperm > 1 and options.mode==2, this is the pvalue for
%       testing the state tiem courses: are these significantly meaningful,
%       above the average accuracy?
% surrogates: if options.Nperm > 1 and options.mode==2, these are the
%       surrogates created by permuting the state time courses across
%       trials
%
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2017)

N = length(T); q = size(Y,2); ttrial = T(1); p = size(X,2);
responses = permute(reshape(Y,[ttrial N q]),[2 3 1]);
responses = responses(:,:,1);
pval = [];  

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
if isfield(options,'Nfeatures') && options.Nfeatures>0 && options.Nfeatures<p
    warning('Specifying Nfeatures can lead to a biased calculation of CV-R2')
end
if isfield(options,'mode') 
    mode = options.mode; options = rmfield(options,'mode');
else, mode = 1; 
end
if isfield(options,'conservative') 
    conservative = options.conservative; options = rmfield(options,'conservative');
else, conservative = 1; 
end
if isfield(options,'estimation_method') 
    estimation_method = options.estimation_method; 
    options = rmfield(options,'estimation_method');
else, estimation_method = 2; 
end
if isfield(options,'Nperm') 
    Nperm = options.Nperm; options = rmfield(options,'Nperm');
else, Nperm = 1; 
end
if isfield(options,'verbose') 
    verbose = options.verbose; options = rmfield(options,'verbose');
else, verbose = 1; 
end
if nargin < 5, Gamma = []; end

    
if ~all(T==T(1)), error('All elements of T must be equal'); end 

% Form CV folds; if response are categorical, then it's stratified
samples_per_value = length(Y) / length(unique(Y));
stratified = samples_per_value > 200;
if ~isfield(options,'c')
    if stratified
        %disp('Response is treated as categorical')
        tmp = zeros(length(responses),1);
        for j = 1:size(responses,2)
            rj = responses(:,j);
            uj = unique(rj);
            for jj = 1:length(uj)
                tmp(rj == uj(jj)) = tmp(rj == uj(jj)) + 100^(j-1) * jj;
            end
        end
        uy = unique(tmp);
        group = zeros(length(responses),1);
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
    c = options.c; options = rmfield(options,'c');
end

% Preproc data and put in the right format
[X,Y,T,options] = preproc4hmm(X,Y,T,options); 
options = remove_options(options);
p = size(X,2); q = size(Y,2);
X = reshape(X,[ttrial N p]);
Y = reshape(Y,[ttrial N q]);
RidgePen = lambda * eye(p);

% Get state time courses
if isempty(Gamma)
    [~,Gamma] = tudatrain(reshape(X,[ttrial*N p]),...
        reshape(Y,[ttrial*N q]),T,options);
end
K = size(Gamma,2); 
Gamma = reshape(Gamma,[ttrial N K]);

% Estimate testing state time courses if necessary
if conservative
    Gammapred = zeros(ttrial,N,K);
    for icv = 1:NCV
        Ntr = sum(c.training{icv}); Nte = sum(c.test{icv});
        Gammatrain = reshape(Gamma(:,c.training{icv},:),[ttrial Ntr K]);
        if estimation_method == 1
            mGammatrain = squeeze(mean(Gammatrain,2));
            for t = 1:ttrial
                [~,k] = max(mGammatrain(t,:));
                Gammapred(t,c.test{icv},k) = 1;
            end
        else
            Xtrain = permute(X(:,c.training{icv},:),[2 3 1]);
            Xtest = permute(X(:,c.test{icv},:),[2 3 1]);
            Gammatrain = permute(reshape(Gamma(:,c.training{icv},:),[ttrial Ntr K]),[2 3 1]);
            for t = 1:ttrial
                B = (Xtrain(:,:,t)' * Xtrain(:,:,t) + 0.0001 * eye(p)) \ ...
                    Xtrain(:,:,t)' * Gammatrain(:,:,t);
                pred = Xtest(:,:,t) * B;
                pred = pred - repmat(min(min(pred,[],2), zeros(Nte,1)),1,K);
                pred = pred ./ repmat(sum(pred,2),1,K);
                Gammapred(t,c.test{icv},:) = pred; 
            end
        end
    end
else
    Gammapred = Gamma;
end

if mode == 1 %  time point by time point CV (i.e. not across time points)
    Ypred = zeros(ttrial,N,q,'single');
    surrogates = zeros(ttrial,Nperm);
    for r = 1:Nperm
        if r==1
            G = Gamma;
            Gp = Gammapred;
        else
            pe = randperm(N,N);
            G = Gamma(:,pe,:);
            Gp = Gammapred(:,pe,:);
        end
        for icv = 1:NCV
            Ntr = sum(c.training{icv}); Nte = sum(c.test{icv});
            Xtrain = reshape(X(:,c.training{icv},:),[ttrial*Ntr p]);
            ytrain = reshape(Y(:,c.training{icv},:),[ttrial*Ntr q]);
            Xtest = reshape(X(:,c.test{icv},:),[ttrial*Nte p]);
            Gammatrain = reshape(G(:,c.training{icv},:),[ttrial*Ntr K]);
            Gammatest = reshape(Gp(:,c.test{icv},:),[ttrial*Nte K]);
            for k = 1:K
                sGamma = repmat(sqrt(Gammatrain(:,k)),1,p);
                Xtrain_k = Xtrain .* sGamma;
                ytrain_k = ytrain .* sGamma(:,1:q);
                Beta = (Xtrain_k' * Xtrain_k + RidgePen) \ (Xtrain_k' * ytrain_k);
                sGamma = repmat(Gammatest(:,k),[1 q]); 
                Ypred(:,c.test{icv},:) = Ypred(:,c.test{icv},:) + ...
                    reshape( (Xtest * Beta) .* sGamma , [ttrial Nte q]);
            end
        end
        if verbose && mod(r,10)==0
            disp(['Permutation ' num2str(r)])
        end
        surrogates(:,r) = get_R2(Y,Ypred,lossfunc);
    end
    R2 = surrogates(:,1);
    if Nperm > 1 
        pval = sum(repmat(R2,[1 Nperm]) <= surrogates ,2) / (Nperm+1);
        surrogates = surrogates(:,2:end);
    else
        surrogates = [];
    end
    
else % cross-time generalisation 
    surrogates = zeros(ttrial,ttrial,Nperm); 
    for r = 1:Nperm
        if r==1
            G = Gamma;
            Gp = Gammapred; 
        else
            pe = randperm(N,N);
            G = Gamma(:,pe,:);
            Gp = Gammapred(:,pe,:);
        end
        for t1 = 1:ttrial
            Ypred = zeros(ttrial,N,q);
            for icv = 1:NCV % This computation could be done just once
                % instead of t1=1:trial, but it would consume lots of memory
                Ntr = sum(c.training{icv}); Nte = sum(c.test{icv});
                Xtrain = reshape(X(:,c.training{icv},:),[ttrial*Ntr p]);
                ytrain = reshape(Y(:,c.training{icv},:),[ttrial*Ntr q]);
                Gammatrain = reshape(G(:,c.training{icv},:),[ttrial*Ntr K]);
                Beta = zeros(p,q,K);
                for k = 1:K
                    % Beta is estimated using all time points in training trials
                    sGamma = repmat(sqrt(Gammatrain(:,k)),1,p);
                    Xtrain_k = Xtrain .* sGamma;
                    ytrain_k = ytrain .* sGamma(:,1:q);
                    Beta(:,:,k) = (Xtrain_k' * Xtrain_k + RidgePen) \ ...
                        (Xtrain_k' * ytrain_k);
                end
                Gammatest = Gp(:,c.test{icv},:);
                nn = find(c.test{icv}');
                for j = 1:length(nn)
                    n = nn(j);
                    % Beta_t1_n is a weighted sum over states,
                    % drawing from Gamma in testing segment, and time point t1
                    Beta_t1_n = zeros(p,q);
                    for k = 1:K
                        Beta_t1_n = Beta_t1_n + Gammatest(t1,j,k) * Beta(:,:,k);
                    end
                    Xtest = permute(X(:,n,:),[1 3 2]); % ttrial by p
                    Ypred(:,n,:) = Xtest * Beta_t1_n;
                end
            end
            surrogates(t1,:,r) = get_R2(Y,Ypred,lossfunc);
        end
        if verbose && mod(r,10)==0
            disp(['Permutation ' num2str(r)])
        end
    end
    R2 = surrogates(:,:,1);
    if Nperm > 1 
        pval = sum(repmat(R2,[1 1 Nperm]) <= surrogates ,3) / (Nperm+1);
        surrogates = surrogates(:,:,2:end);
    else
        surrogates = [];
    end
    
end

end


function options = remove_options(options)
% things that we don't want tudatrain to do again
if isfield(options,'filter'), options = rmfield(options,'filter'); end
if isfield(options,'Nfeatures'), options = rmfield(options,'Nfeatures'); end
if isfield(options,'detrend'), options = rmfield(options,'detrend'); end
if isfield(options,'onpower'), options = rmfield(options,'onpower'); end
if isfield(options,'standardise'), options = rmfield(options,'standardise'); end
if isfield(options,'embeddedlags'), options = rmfield(options,'embeddedlags'); end
if isfield(options,'pca'), options = rmfield(options,'pca'); end
options.add_noise = 0; 
end


function R2 = get_R2(Y,Ypred,lossfunc)
d = Y - Ypred;
if strcmp(lossfunc,'quadratic')
    l = d.^2; l0 = Y.^2; ee = 1/2;
elseif strcmp(lossfunc,'absolute')
    l = abs(d); l0 = abs(Y); ee = 1;
elseif strcmp(lossfunc,'huber')
    l = zeros(size(d)); l0 = zeros(size(d)); ee = 1;
    for j1 = 1:N
        for j2 = 1:q
            ii = abs(d(:,j1,j2))<1; l(ii,j1,j2) = d(ii,j1,j2).^2;
            ii = abs(d(:,j1,j2))>=1; l(ii,j1,j2) = abs(d(ii,j1,j2));
            ii = abs(Y(:,j1,j2))<1; l0(ii,j1,j2) = Y(ii,j1,j2).^2;
            ii = abs(Y(:,j1,j2))>=1; l0(ii,j1,j2) = abs(Y(ii,j1,j2));
        end
    end
end
% across-trial R2, using mean of euclidean distances in stimulus space
m = mean(sum(l,3).^(ee),2);
m0 = mean(sum(l0,3).^(ee),2);
R2 = 1 - m ./ m0;
% % mean of R2, one per trial - equivalent to the previous
% m = sum(l,3).^(ee);
% m0 = sum(l0,3).^(ee);
% R2 = mean(1 - m ./ m0,2);
% % computing SE with all trials and features at once
% se = sum(sum(l,3),2).^(ee);
% se0 = sum(sum(l0,3),2).^(ee);
% R2 = 1 - se ./ se0;
end
