function [acc,acc_star,Ypred,Ypred_star] = tucacv(X,Y,T,options)
%
% Performs cross-validation of the TUCA model, which can be useful for
% example to compare different number or states or other parameters
%
% 
%
% INPUT
%
% X: Brain data, (time by regions)
% Y: Stimulus, (time by q); q is no. of stimulus features
%               For binary classification problems, Y is (time by 1) and
%               has values -1 or 1
%               For multiclass classification problems, Y is (time by classes) 
%               with indicators values taking 0 or 1. 
%           If the stimulus is the same for all trials, Y can have as many
%           rows as trials, e.g. (trials by q) 
% T: Length of series or trials
% options: structure with the training options - see documentation in
%                       https://github.com/OHBA-analysis/HMM-MAR/wiki
%  Apart from the options specified for tudatrain, these are specific to tudacv:
%  - options.CVmethod, This options establishes how to compute the model time
%                  courses in the held-out data. Note that it is not
%                  obvious which state to use in the held-out data, because
%                  deciding which one is the most appropriate needs to use Y,
%                  which is precisely what we aim to predict. Ways of
%                  estimating it in a non-circular way are:
%                  . options.CVmethod=1: the state time course in held-out trials 
%                  is taken to be the average from training. That is, if 20% of
%                  the training trials use decoder 1, and 80% use decoder 2,
%                  then the prediction in testing will be a weighted average of these  
%                  two decoders, with weights 0.8 and 0.2. 
%                  . options.CVmethod=2, the state time courses in testing are estimated 
%                  using just data and linear regression, i.e. we try to predict
%                  the state time courses in held-out trials using the data
%  - options.NCV, containing the number of cross-validation folds (default 10)
%  - options.lambda, regularisation penalty for estimating the testing
%  state time courses when options.CVmethod=2.
%  - options.c      an optional CV fold structure as returned by cvpartition
%
% OUTPUT
%
% acc: cross-validated accuracy ? explained variance if Y is continuous,
%           classification accuracy if Y is categorical (one value)
% acc_star: cross-validated accuracy across time (trial time by 1) 
% Ypred: predicted stimulus (trials by stimuli/classes)
% Ypred_star: predicted stimulus across time (time by trials by stimuli/classes)
%
% Author: Cam Higgins, OHBA, University of Oxford (2018)

if ~isfield(options,'classifier')
    options.classifier='logistic'; %set default
end
classifier=options.classifier;
do_preproc = 1; 

N = length(T); q = size(Y,2); ttrial = T(1); p = size(X,2); K = options.K;
if ~all(T==T(1)), error('All elements of T must be equal for cross validation'); end 

if size(Y,1) == length(T) % one value per trial
    responses = Y;
    Ystar = reshape(repmat(reshape(responses,[1 N q]),[ttrial,1,1]),[ttrial*N q]);
else
    responses = reshape(Y,[ttrial N q]);
    Ystar = reshape(repmat(responses(1,:,:),[ttrial,1,1]),[ttrial*N q]);
    responses = permute(responses(1,:,:),[2 3 1]); % N x q
    if ~all(Y(:)==Ystar(:))
       error('For cross-validating, the same stimulus must be presented for the entire trial');
    end
end

Ycopy = Y;
if size(Ycopy,1) == N 
    Ycopy = repmat(reshape(Ycopy,[1 N q]),[ttrial 1 1]);
end
% no demeaning by default if this is a classification problem
if ~isfield(options,'demeanstim'), options.demeanstim = 0; end


% Preproc data and put in the right format 
if do_preproc
    if isfield(options,'embeddedlags'), el = options.embeddedlags;else;el=0;end
    [X,Y,T,options] = preproc4hmm(X,Y,T,options); options.classifier=classifier;
    p = size(X,2);Q_star=size(Y,2);
    if Q_star~=q && strcmp(options.distribution,'logistic')
        Ycopy = multinomToBinary(Ycopy);
        q=size(Ycopy,2);
    end
    if length(el) > 1
        Ycopy = reshape(Ycopy,[ttrial N q]);
        Ycopy = Ycopy(-el(1)+1:end-el(end),:,:);
        Ycopy = reshape(Ycopy,[T(1)*N q]);
    end
    ttrial = T(1); 
    if strcmp(classifier,'LDA')
        options.intercept=false; %this necessary to avoid double addition of intercept terms
    end
end

if isfield(options,'CVmethod') 
    CVmethod = options.CVmethod; options = rmfield(options,'CVmethod');
else
    CVmethod = 1;
end
class_totals=(sum(Y==1)./ttrial);
if Q_star==(q+1);class_totals=class_totals(2:end);end %remove intercepte term
if size(unique(class_totals))>1
    warning('Note that Y is not balanced; cross validation folds will not be balanced and predictions will be biased')
end
if isfield(options,'c')
    NCV = options.c.NumTestSets;
    if isfield(options,'NCV');options = rmfield(options,'NCV');end
elseif isfield(options,'NCV')
    NCV = options.NCV; 
    options = rmfield(options,'NCV');
else
    %default to hold one-out CV unless NCV>20:
    NCV = max(class_totals);
    if NCV>20;NCV=20;end
end

if isfield(options,'lambda') 
    lambda = options.lambda; options = rmfield(options,'lambda');
else, lambda = 0.0001; 
end
if isfield(options,'Nfeatures') && options.Nfeatures>0 && options.Nfeatures<p
    error('Specifying Nfeatures can lead to a biased calculation of CV-R2')
end
if isfield(options,'verbose') 
    verbose = options.verbose; options = rmfield(options,'verbose');
else, verbose = 1; 
end

if ~isfield(options,'c')
    %disp('Response is treated as categorical')
    tmp = zeros(N,1);
    for j = 1:size(responses,2)
        rj = responses(:,j);
        uj = unique(rj);
        for jj = 1:length(uj)
            tmp(rj == uj(jj)) = tmp(rj == uj(jj)) + 100^(j-1) * jj;
        end
    end
    uy = unique(tmp);
    group = zeros(N,1);
    for j = 1:length(uy)
        group(tmp == uy(j)) = j;
    end
    c2 = cvpartition(group,'KFold',NCV);
else
   c2 = options.c; options = rmfield(options,'c');
end
c = struct();
c.test = cell(NCV,1);
c.training = cell(NCV,1);
for icv = 1:NCV
    c.training{icv} = find(c2.training(icv));
    c.test{icv} = find(c2.test(icv));
end; clear c2

X = reshape(X,[ttrial N p]);
Y = reshape(Y,[ttrial N Q_star]);
RidgePen = lambda * eye(p);

% Get Gamma and the Betas for each fold
Gammapred = zeros(ttrial,N,K); Betas = zeros(p,Q_star,K,NCV); 
for icv = 1:NCV
    Ntr = length(c.training{icv}); Nte = length(c.test{icv});
    Xtrain = reshape(X(:,c.training{icv},:),[Ntr*ttrial p] ) ;
    Ytrain = reshape(Y(:,c.training{icv},:),[Ntr*ttrial Q_star] ) ;
    Ttr = T(c.training{icv});
    [tuca,Gammatrain] = tucatrain(Xtrain,Ytrain,Ttr,options);
    if strcmp(classifier,'logistic')
        Betas(:,:,:,icv) = tudabeta(tuca);
    elseif strcmp(classifier,'LDA')
        LDAmodel{icv} = tuca;
    end
    switch CVmethod
        case 1 % training average
            mGammatrain = squeeze(mean(reshape(Gammatrain,[ttrial Ntr K]),2)); 
            for j = 1:Nte, Gammapred(:,c.test{icv}(j),:) = mGammatrain; end
        case 2 % regression
            Xtest = permute(X(:,c.test{icv},:),[2 3 1]);
            Xtrain = permute(X(:,c.training{icv},:),[2 3 1]);
            Gammatrain = permute(reshape(Gammatrain,[ttrial Ntr K]),[2 3 1]);
            for t = 1:ttrial
                B = (Xtrain(:,:,t)' * Xtrain(:,:,t) + RidgePen) \ ...
                    Xtrain(:,:,t)' * Gammatrain(:,:,t);
                pred = Xtest(:,:,t) * B;
                pred = pred - repmat(min(min(pred,[],2), zeros(Nte,1)),1,K);
                pred = pred ./ repmat(sum(pred,2),1,K);
                Gammapred(t,c.test{icv},:) = pred;
            end
    end      
    if verbose, fprintf(['\nCV iteration: ' num2str(icv),' of ',int2str(NCV),'\n']); end
end

% Perform the prediction 
if strcmp(classifier,'logistic')
    Ypred = zeros(ttrial,N,Q_star);
elseif strcmp(classifier,'LDA')
    Ypred = zeros(ttrial,N,q);
end
for icv = 1:NCV
    Nte = length(c.test{icv});
    Xtest = reshape(X(:,c.test{icv},:),[ttrial*Nte p]);
    Gammatest = reshape(Gammapred(:,c.test{icv},:),[ttrial*Nte K]);
    if strcmp(classifier,'logistic')
        for k = 1:K
            sGamma = repmat(Gammatest(:,k),[1 Q_star]);
            Ypred(:,c.test{icv},:) = Ypred(:,c.test{icv},:) + ...
                reshape( (Xtest * Betas(:,:,k,icv)) .* sGamma , [ttrial Nte Q_star]);
        end
    elseif strcmp(classifier,'LDA')
         predictions = LDApredict(LDAmodel{icv},Gammatest,Xtest);
         Ypred(:,c.test{icv},:) = reshape(predictions,[ttrial Nte q]);
    end
end
if strcmp(options.distribution,'logistic')
    if length(unique(responses))==2 % denotes binary logistic regression
        Ypred = log_sigmoid(Ypred);
    else %multivariate logistic regression
        Ypred = multinomLogRegPred(Ypred);
    end
end

% and compute accuracy metrics:
Y = reshape(Ycopy,[ttrial*N q]);
Y = continuous_prediction_2class(Ycopy,Y); % get rid of noise we might have injected 

Ypred = reshape(Ypred,[ttrial*N q]);
Ypred_star = reshape(continuous_prediction_2class(Ycopy,Ypred),[ttrial N q]); % this implements a hardmax classification
Ypred = zeros(N,q); 
for j = 1:N % getting the most likely class for all time points in trial
    if q == 1 % binary classification, -1 vs 1
        Ypred(j) = sign(mean(Ypred_star(:,j,1)));
    else
       [~,cl] = max(mean(permute(Ypred_star(:,j,:),[1 3 2])));
       Ypred(j,cl) = 1; 
    end
end
% acc is cross-validated classification accuracy 
Ypred_star = reshape(Ypred_star,[ttrial*N q]);
if q == 1
    tmp = abs(Y - Ypred_star) < 1e-4;
else
    % require all classes in a row to be correct:
    tmp = sum(abs(Y - Ypred_star),2) < 1e-4;
end
acc = mean(tmp);
acc_star = squeeze(mean(reshape(tmp, [ttrial N 1]),2));

    
end

function Y_out = multinomToBinary(Y_in)
Y_out=zeros(length(Y_in),length(unique(Y_in)));
for i=1:length(Y_in)
    Y_out(i,Y_in(i))=1;
end
end