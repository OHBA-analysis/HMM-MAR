function [R2,c] = tudacv(X,Y,T,options)
% Performs cross-validation of the TUDA model, which can be useful for
% example to compare different number or states or other parameters
% Cross-validation has a limitation for TUDA that is acknowledged in
%   Vidaurre et al (2017).Spontaneous	cortical activity transiently organises 
%       into frequency specific phase-coupling networks
%   (section 'Modelling between-trial temporal differences improves decoding
%   performance')
% Specificying Nfeatures will lead to a circular assessment (overfitting of
% R2), so this is discouraged
% 
% INPUT
% X: Brain data, (time by regions)
% Y: Stimulus, (time by q); q is no. of stimulus features
% T: Length of series
% options: structure with the training options - see documentation in 
%                       https://github.com/OHBA-analysis/HMM-MAR/wiki
%  Apart from the options specified for tudatrain, specific to tudacv are:
%  - options.NCV, containing the number of cross-validation folds (default 10)
%  - options.lambda, regularisation penalty for the decoding
%  - options.lossfunc, loss function to compute the cross-validated error, 
%           default is 'quadratic', in which case R2 corresponds to
%           explained variance. Other options are: 'absolute', 'huber'
%
% OUTPUT 
% R2: cross-validated explained variance
% c: CV folds structure (c.training, c.test)
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2017)

N = length(T); q = size(Y,2); ttrial = T(1); p = size(X,2);
responses = permute(reshape(Y,[ttrial N q]),[2 3 1]);
responses = responses(:,:,1);

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
    
if ~all(T==T(1)), error('All elements of T must be equal'); end 

% Form CV folds; if response are categorical, then it's stratified
samples_per_value = length(Y) / length(unique(Y));
stratified = samples_per_value > 500;
if stratified
    disp('Response is treated as categorical')
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
    disp('Response is treated as continuous - no CV stratification')
end

c = struct();
c.test = cell(NCV,1);
c.training = cell(NCV,1);
for icv = 1:NCV
    c.training{icv} = c2.training(icv);
    c.test{icv} = c2.test(icv);
end; clear c2
K = options.K;

% Preproc data and put in the right format
[X,Y,T,options] = preproc4hmm(X,Y,T,options); 
options = remove_options(options);
p = size(X,2); q = size(p,2);
X = reshape(X,[ttrial N p]);
Y = reshape(Y,[ttrial N q]);
Ypred = zeros(ttrial,N,q,'single');

RidgePen = lambda * eye(p);

for icv = 1:NCV
    Ntr = sum(c.training{icv}); Nte = sum(c.test{icv});
    Xtrain = reshape(X(:,c.training{icv},:),[ttrial*Ntr p]);
    ytrain = reshape(Y(:,c.training{icv},:),[ttrial*Ntr q]);
    Xtest = reshape(X(:,c.test{icv},:),[ttrial*Nte p]);
    [~,Gammatrain] = tudatrain(Xtrain,ytrain,T(c.training{icv}),options);
    mGammatrain = squeeze(mean(reshape(Gammatrain,[ttrial Ntr K]),2));
    Gammatest = zeros(ttrial,K);
    for t = 1:ttrial
        [~,k] = max(mGammatrain(t,:));
        Gammatest(t,k) = 1;  
    end
    for k = 1:K
        sGamma = repmat(sqrt(Gammatrain(:,k)),1,p); 
        Xtrain_k = Xtrain .* sGamma;
        ytrain_k = ytrain .* sGamma(:,1:q);
        Beta = (Xtrain_k' * Xtrain_k + RidgePen) \ (Xtrain_k' * ytrain_k);
        sGamma = reshape(repmat(Gammatest(:,k),[1,Nte,q]),[ttrial*Nte q]); 
        Ypred(:,c.test{icv},:) = Ypred(:,c.test{icv},:) + ...
            reshape( (Xtest * Beta) .* sGamma , [ttrial Nte q]);  
    end
end

d = Y - Ypred;
if strcmp(lossfunc,'quadratic')
    l = d.^2; l0 = Y.^2; ee = 1/2;
elseif strcmp(lossfunc,'absolute')
    l = abs(d); l0 = abs(Y); ee = 1;
elseif strcmp(lossfunc,'huber')
    l = zeros(size(d)); l0 = zeros(size(d)); ee = 1;
    for j1 = 1:N
        for j = 1:q
            ii = abs(d(:,j1,j2))<1; l(ii,j1,j2) = d(ii,j1,j2).^2;
            ii = abs(d(:,j1,j2))>=1; l(ii,j1,j2) = abs(d(ii,j1,j2));
            ii = abs(Y(:,j1,j2))<1; l0(ii,j1,j2) = Y(ii,j1,j2).^2;
            ii = abs(Y(:,j1,j2))>=1; l0(ii,j1,j2) = abs(Y(ii,j1,j2));
        end
    end
end
 
m = mean(sum(l,3).^(ee),2);
m0 = mean(sum(l0,3).^(ee),2);
R2 = 1 - m ./ m0; 

% R2 time point by time point is insane
% R2M = 1 - SE ./ SE0;
% R2 = struct(); 
% R2.m = mean(R2M,2); % ttrial by 1
% R2.s = std(R2M,[],2); % ttrial by 1

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
end

