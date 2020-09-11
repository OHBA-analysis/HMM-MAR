function [acc,acc_star,Ypred,Ypred_star,Gammapred,acc_Gamma] = tudacv(X,Y,T,options)
%
% Performs cross-validation of the TUDA model, which can be useful for
% example to compare different number or states or other parameters
% (the words decoder and state are used below indistinctly)
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
% acc: cross-validated explained variance if Y is continuous,
%           classification accuracy if Y is categorical (one value)
% acc_star: cross-validated accuracy across time (trial time by 1) 
% Ypred: predicted stimulus (trials by stimuli/classes)
% Ypred_star: (soft) predicted stimulus across time (time by trials by stimuli/classes)
% Gammapred: the predicted state timecourses used on the held out data set.
%       Note that unbiased testing requires a secondary estimation of the
%       state timecourses, so these may deviate from slightly from the true
%       model state timecourses and are returned for error checking.
% acc_Gamma: the decoder accuracy across all trials as a function of the 
%       active state.
%
% Author: Diego Vidaurre, OHBA, University of Oxford 
% Author: Cam Higgins, OHBA, University of Oxford  

N = length(T); q = size(Y,2); ttrial = T(1); K = options.K;
if ~all(T==T(1)), error('All elements of T must be equal for cross validation'); end 

if size(Y,1) == length(T) % one value per trial
    responses = Y;
else
    responses = reshape(Y,[ttrial N q]);
    responses = permute(responses(1,:,:),[2 3 1]); % N x q
end

options.Nfeatures = 0;
Ycopy = Y;
if size(Ycopy,1) == N 
    Ycopy = repmat(reshape(Ycopy,[1 N q]),[ttrial 1 1]);
end
[X,Y,T,options] = preproc4hmm(X,Y,T,options); % this demeans Y if necessary
ttrial = T(1); p = size(X,2); q_star = size(Y,2);
classifier = options.classifier;
classification = ~isempty(classifier);
if classification, Ycopy = round(Ycopy); end
if q_star~=q && strcmp(options.distribution,'logistic')
    Ycopy = multinomToBinary(Ycopy);
    q = size(Ycopy,2);   
    responses = Ycopy(cumsum(T),:);
end
if strcmp(classifier,'LDA') || options.encodemodel
    options.intercept = false; %this necessary to avoid double addition of intercept terms
end

if isfield(options,'CVmethod') 
    CVmethod = options.CVmethod; options = rmfield(options,'CVmethod');
else
    CVmethod = 1;
end
class_totals = (sum(Ycopy==1)./ttrial);
if q_star == (q+1)
    class_totals = class_totals(2:end); %remove intercept term
end 
if size(unique(class_totals))>1
    warning(['Note that Y is not balanced; ' ...
        'cross validation folds will not be balanced and predictions will be biased'])
end
if isfield(options,'c')
    NCV = options.c.NumTestSets;
    if isfield(options,'NCV'), options = rmfield(options,'NCV'); end
elseif isfield(options,'NCV')
    NCV = options.NCV; 
    options = rmfield(options,'NCV');
else
    %default to hold one-out CV unless NCV>10:
    NCV = max([0,class_totals]);
    if NCV > 10 || NCV < 1, NCV = 10; end
    
end
if isfield(options,'lambda') 
    lambda = options.lambda; options = rmfield(options,'lambda');
else, lambda = 0.0001; 
end
if isfield(options,'verbose') 
    verbose = options.verbose; options = rmfield(options,'verbose');
else, verbose = 1; 
end
if isfield(options,'accuracyType')
    accuracyType = options.accuracyType;
    options = rmfield(options,'accuracyType');
else
    accuracyType = 'COD';
end
options.verbose = 0; 

if ~isfield(options,'c')
    % this system is thought for cases where a trial can have more than 
    % 1 category, and potentially each column can have more than 2 values,
    % but there are not too many categories
    if classification 
        tmp = zeros(N,1);
        for j = 1:q
            rj = responses(:,j);
            uj = unique(rj);
            for jj = 1:length(uj)
                tmp(rj == uj(jj)) = tmp(rj == uj(jj)) + (q+1)^(j-1) * jj;
            end
        end
        uy = unique(tmp);
        group = zeros(N,1);
        for j = 1:length(uy)
            group(tmp == uy(j)) = j;
        end
        c2 = cvpartition(group,'KFold',NCV);
    else % Response is treated as continuous - no CV stratification
        c2 = cvpartition(N,'KFold',NCV);
    end
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
Y = reshape(Y,[ttrial N q_star]);

% Get Gamma and the Betas for each fold
Gammapred = zeros(ttrial,N,K,length(CVmethod)); 
Betas = zeros(p,q_star,K,NCV); 
if strcmp(classifier,'LDA')
    LDAmodel = cell(NCV,1);
end
if strcmp(options.classifier,'regression'), options.classifier = ''; end
if options.encodemodel
    options.classifier = 'LDA';
    classifier = 'LDA';
    options = rmfield(options,'encodemodel');
end 
for icv = 1:NCV
    Ntr = length(c.training{icv}); Nte = length(c.test{icv});
    Xtrain = reshape(X(:,c.training{icv},:),[Ntr*ttrial p] ) ;
    Ytrain = reshape(Y(:,c.training{icv},:),[Ntr*ttrial q_star] ) ;
    Ttr = T(c.training{icv});
    [tuda,Gammatrain] = tudatrain(Xtrain,Ytrain,Ttr,options);
    if strcmp(classifier,'LDA')
        LDAmodel{icv} = tuda;
    else
        Betas(:,:,:,icv) = tudabeta(tuda);
    end   
    for iFitMethod = 1:length(CVmethod)
        iCVm = CVmethod(iFitMethod);
        switch iCVm
            case 1 % training average
                mGammatrain = squeeze(mean(reshape(Gammatrain,[ttrial Ntr K]),2)); 
                for j = 1:Nte, Gammapred(:,c.test{icv}(j),:,iFitMethod) = mGammatrain; end
    
            case 2
                % regression
                Xtest2 = permute(X(:,c.test{icv},:),[2 3 1]);
                Xtrain2 = permute(X(:,c.training{icv},:),[2 3 1]);
                Xtest2 = cat(2,Xtest2,ones(Nte,1,ttrial)); %include intercept term
                Xtrain2 = cat(2,Xtrain2,ones(Ntr,1,ttrial));
                RidgePen = lambda * eye(p+1);
                Gammatrain2 = permute(reshape(Gammatrain,[ttrial Ntr K]),[2 3 1]);
                for t = 1:ttrial
                    B = (Xtrain2(:,:,t)' * Xtrain2(:,:,t) + RidgePen) \ ...
                        Xtrain2(:,:,t)' * Gammatrain2(:,:,t);
                    pred = Xtest2(:,:,t) * B;
                    pred = pred - repmat(min(min(pred,[],2), zeros(Nte,1)),1,K);
                    pred = pred ./ repmat(sum(pred,2),1,K);
                    Gammapred(t,c.test{icv},:,iFitMethod) = pred;
                end
            case 3
                % distributional model
                Xtrain2 = reshape(X(:,c.training{icv},:),[ttrial*length(c.training{icv}),p]);
                Xtest2 = reshape(X(:,c.test{icv},:),[ttrial*length(c.test{icv}),p]);
                GammaTemp = fitEquivUnsupervisedModel(Xtrain2,Gammatrain,Xtest2,T(c.training{icv}),T(c.test{icv}));
                Gammapred(:,c.test{icv},:,iFitMethod) = reshape(GammaTemp,[ttrial,length(c.test{icv}),K]);
            case 4
                % best model (subject to overfitting)
                Xtest2 = reshape(X(:,c.test{icv},:),[ttrial*length(c.test{icv}),p]);
                Ytest2 = reshape(Y(:,c.test{icv},:),[ttrial*length(c.test{icv}) q_star] ) ;
                Tte = T(c.test{icv});
                Gammapred(:,c.test{icv},:,iFitMethod) = reshape(tudadecode(Xtest2,Ytest2,Tte,tuda),[ttrial Nte K]);
            case 5
                % estimate MLE for Y and then state timecourses:
                Xtest2 = reshape(X(:,c.test{icv},:),[ttrial*length(c.test{icv}),p]);
                Tte = T(c.test{icv});
                mGammatest = squeeze(repmat(mean(reshape(Gammatrain,[ttrial Ntr K]),2),[Nte,1]));
                [Y_test_MLE,Y_test_LL] = LDApredict(LDAmodel{icv},mGammatest,Xtest2,classification,var(Ytrain(:,1))==0);
                if classification % use soft probabilities:
                    %Y_test_MLE = exp(Y_test_LL - repmat(max(Y_test_LL,[],2),1,K));
                    %Y_test_MLE = rdiv(Y_test_MLE,sum(Y_test_MLE,2));
                end
                if q_star>q,Y_test_MLE = [ones(sum(Tte),1),Y_test_MLE];end
                Gammapred(:,c.test{icv},:,iFitMethod) = reshape(tudadecode(Xtest2,Y_test_MLE,Tte,tuda),[ttrial Nte K]);
            
        end
    end
    if verbose
        fprintf(['\nCV iteration: ' num2str(icv),' of ',int2str(NCV),'\n'])
    end
end

% Perform the prediction 
if strcmp(classifier,'LDA')
    Ypred = zeros(ttrial,N,q,length(CVmethod));
else
    Ypred = zeros(ttrial,N,q_star,length(CVmethod));
end
nCVm = length(CVmethod);
for iFitMethod = 1:nCVm
    for icv = 1:NCV
        Nte = length(c.test{icv});
        Xtest = reshape(X(:,c.test{icv},:),[ttrial*Nte p]);
        Gammatest = reshape(Gammapred(:,c.test{icv},:,iFitMethod),[ttrial*Nte K]);
        if strcmp(classifier,'LDA')
            [predictions,predictions_soft] = LDApredict(LDAmodel{icv},Gammatest,Xtest,classification,var(Ytrain(:,1))==0);
            predictions_soft = exp(predictions_soft - repmat(max(predictions_soft,[],2),1,q));
            predictions_soft = rdiv(predictions_soft,sum(predictions_soft,2)); 
            Ypred(:,c.test{icv},:,iFitMethod) = reshape(predictions_soft,[ttrial Nte q]);
        else 
            for k = 1:K
                sGamma = repmat(Gammatest(:,k),[1 q_star]);
                Ypred(:,c.test{icv},:,iFitMethod) = Ypred(:,c.test{icv},:,iFitMethod) + ...
                    reshape( (Xtest * Betas(:,:,k,icv)) .* sGamma , [ttrial Nte q_star]);
            end
        end
    end
end
if strcmp(options.distribution,'logistic')
    if q_star==q % denotes binary logistic regression
        Ypred = log_sigmoid(Ypred);
    else %multivariate logistic regression
        for i=1:size(Ypred,4)
            Ypredtemp(:,:,:,i) = multinomLogRegPred(Ypred(:,:,:,i));
        end
        Ypred = Ypredtemp;
    end
end
acc_Gamma = zeros(K,nCVm);
for iFitMethod = 1:nCVm
    if classification
        Y = reshape(Ycopy,[ttrial*N q]);
        Y = continuous_prediction_2class(Ycopy,Y); % get rid of noise we might have injected 
        Ypred_temp = reshape(Ypred(:,:,:,iFitMethod),[ttrial*N q]);
        Ypred_star_temp = reshape(continuous_prediction_2class(Ycopy,Ypred_temp),[ttrial N q]);
        Ypred_temp = zeros(N,q); 
        for j = 1:N % getting the most likely class for all time points in trial
            if q == 1 % binary classification, -1 vs 1
                Ypred_temp(j) = sign(mean(Ypred_star_temp(:,j,1)));
            else
               [~,cl] = max(mean(permute(Ypred_star_temp(:,j,:),[1 3 2])));
               Ypred_temp(j,cl) = 1; 
            end
        end
        % acc is cross-validated classification accuracy 
        Ypred_star_temp = reshape(Ypred_star_temp,[ttrial*N q]);
        if q == 1
            tmp = abs(Y - Ypred_star_temp) < 1e-4;
        else
            tmp = sum(abs(Y - Ypred_star_temp),2) < 1e-4;
        end
        acc_temp = mean(tmp);
        acc_star_temp = squeeze(mean(reshape(tmp, [ttrial N 1]),2));
        if nargout==6
            Gammapredtemp = reshape(Gammapred(:,:,:,iFitMethod),[ttrial*N K]);
            for iK=1:K
                acc_Gamma(iK,iFitMethod) = sum(tmp.*Gammapredtemp(:,iK)) ./ sum(Gammapredtemp(:,iK));
            end
        end    
    else
        acc_Gamma = zeros(K,q,nCVm);
        Y = reshape(Ycopy,[ttrial*N q]);
        Ypred_star_temp =  reshape(Ypred(:,:,:,iFitMethod), [ttrial*N q]); 
        Ypred_temp = permute( mean(Ypred(:,:,:,iFitMethod),1) ,[2 3 1]);
        if strcmp(accuracyType,'COD')
            % acc is explained variance 
            acc_temp = 1 - sum( (Y - Ypred_star_temp).^2 ) ./ sum(Y.^2) ; 
            if nargout==6
                Gammapredtemp = reshape(Gammapred(:,:,:,iFitMethod),[ttrial*N K]);
                GammaMLE = Gammapredtemp==repmat(max(Gammapredtemp,[],2),1,K);
                if all(sum(GammaMLE)>10)
                    % use hard state assignments unless insufficient samples:
                    Gammapredtemp = GammaMLE;
                end
                for iK=1:K
                    acc_Gamma(iK,:,iFitMethod) = diag(1 - weighted_covariance(Y - Ypred_star_temp,Gammapredtemp(:,iK)) ./ weighted_covariance(Y,Gammapredtemp(:,iK)));
                end
            end
        elseif strcmp(accuracyType,'Pearson')
            acc_temp = diag(corr(Y,Ypred_star_temp));
            if nargout==6
                Gammapredtemp = reshape(Gammapred(:,:,:,iFitMethod),[ttrial*N K]);
                GammaMLE = Gammapredtemp==repmat(max(Gammapredtemp,[],2),1,K);
                if all(sum(GammaMLE)>10)
                    % use hard state assignments unless insufficient samples:
                    Gammapredtemp = GammaMLE;
                end
                for iK=1:K
                    Ctemp = weighted_covariance([Y,Ypred_star_temp],Gammapredtemp(:,iK));
                    for ireg=1:q
                        acc_Gamma(iK,ireg,iFitMethod) = Ctemp(q+ireg,ireg)./sqrt(prod([Ctemp(ireg,ireg),Ctemp(q+ireg,q+ireg)]));
                    end
                end
            end
        end
        acc_star_temp = zeros(ttrial,q); 
        Y = reshape(Y,[ttrial N q]);
        Ypred_star_temp = reshape(Ypred_star_temp, [ttrial N q]);
        for t = 1:ttrial
            y = permute(Y(t,:,:),[2 3 1]); 
            if strcmp(accuracyType,'COD')
                acc_star_temp(t,:) = 1 - sum((y - permute(Ypred_star_temp(t,:,:),[2 3 1])).^2) ./ sum(y.^2);
            elseif strcmp(accuracyType,'Pearson')
                acc_star_temp(t,:) = diag(corr(y,permute(Ypred_star_temp(t,:,:),[2 3 1])));
                
            end
        end
        Ypred_star_temp = reshape(Ypred_star_temp, [ttrial*N q]);
    end
    acc(:,iFitMethod) = acc_temp;
    acc_star(:,:,iFitMethod) = acc_star_temp;
    Ypred_out(:,:,iFitMethod) = Ypred_temp;
    Ypred_star(:,:,iFitMethod) = Ypred_star_temp;
end
Ypred_star = reshape(Ypred,[ttrial,N,q,length(CVmethod)]);
Ypred = Ypred_out;
Gammapred = reshape(Gammapred,[ttrial,N,K,length(CVmethod)]); 

end



function Y_out = multinomToBinary(Y_in)
Y_out=zeros(length(Y_in),length(unique(Y_in)));
for i=1:length(Y_in)
    Y_out(i,Y_in(i))=1;
end
end

