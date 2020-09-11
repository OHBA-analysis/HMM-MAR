function [cv_acc,acc,meanGenPlot,AUC,LL] = standard_classification(X,Y,T,options,binsize)
% Determine the cross validated accuracy of a classification type on brain
% data X with mutually exclusive classes Y
%
% INPUT
% X: Brain data, (total time by regions) or (time by trials by regions)
% Y: Stimulus, (total time by q). This variable can either be entered as a
%       single vector of class assignments (ie q=1 and y takes values
%       1,2,...N=total number of classes); or can be entered as a matrix of
%       labels where each column contains 1 or 0 denoting whether that
%       class is active.
% T: Length of series
%       options             structure with the following subfields:
%           classifier      String indicating classifier type;
%                   logistic    Logistic Regression classifier (default)
%                   SVM         Linear support vector machine classifier
%                   LDA         Linear Discriminant Analysis classifier
%           regularisation  String indicating type of regularisation;
%                   L1          Lasso regularisation
%                   L2          Ridge regularisation (default)
%                   ARD         Automatic relevance determination
%           lambda          Vector of regularisation penalty values to be
%                           tested
% binsize: NOT YET IMPLEMENTED 
%           how many consecutive time points will be used for each estiamtion. 
%           By default, this is 1, such that one decoding model
%               is estimated using 1 time point
%
% OUTPUT
% 
%
% Author: Cam Higgins, OHBA, University of Oxford (2019)

if ~isfield(options,'classifier')
    options.classifier = 'logistic'; %set default
end
classifier = options.classifier;
do_preproc = 1; 

N = length(T); q = size(Y,2); ttrial = T(1); p = size(X,2); 
if ~all(T==T(1)), error('All elements of T must be equal for cross validation'); end 

if size(Y,1) == length(T) % one value per trial
    responses = Y;
else
    responses = reshape(Y,[ttrial N q]);
    Ytemp = reshape(repmat(responses(1,:,:),[ttrial,1,1]),[ttrial*N q]);
    responses = permute(responses(1,:,:),[2 3 1]); % N x q
    if ~all(Y(:)==Ytemp(:))
       error('For cross-validating, the same stimulus must be presented for the entire trial');
    end
end

Ycopy = Y;
if size(Ycopy,1) == N 
    Ycopy = repmat(reshape(Ycopy,[1 N q]),[ttrial 1 1]);
end
% no demeaning by default if this is a classification problem
if ~isfield(options,'demeanstim'), options.demeanstim = 0; end
if ~isfield(options,'verbose'),options.verbose=0;verbose_CV=1;
else,verbose_CV=options.verbose;
end

% Preproc data and put in the right format 
if do_preproc
    if isfield(options,'embeddedlags'), el = options.embeddedlags;else;el=0;end
    options.K=0;
    [X,Y,T,options] = preproc4hmm(X,Y,T,options); 
    p = size(X,2);Q_star=size(Y,2);
    if Q_star~=q && strcmp(options.distribution,'logistic')
        Ycopy = multinomToBinary(Ycopy);
        q = size(Ycopy,2);
        responses = Ycopy(cumsum(T),:);
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

class_totals = (sum(Y==1)./ttrial);
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

if ~isfield(options,'c') 
    % this system is thought for cases where a trial can have more than
    % 1 category, and potentially each column can have more than 2 values,
    % but there are not too many categories
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
Ycopy = reshape(Ycopy,[ttrial N q]);

% Fit classifier to each fold:
cv_acc = zeros(ttrial,NCV);
LL = zeros(ttrial,N);
for icv = 1:NCV
    % train classifier on training set:
    Ntr = length(c.training{icv}); 
    Xtrain = reshape(X(:,c.training{icv},:),[Ntr*ttrial p] ) ;
    Ytrain = reshape(Y(:,c.training{icv},:),[Ntr*ttrial Q_star] ) ;
    Ttr = T(c.training{icv});
    model = standard_classifier_train(Xtrain,Ytrain,Ttr,options);
    
    % and test on test set:
    Nte = length(c.test{icv});
    Xtest = reshape(X(:,c.test{icv},:),[ttrial*Nte p]);
    Ytest = reshape(Ycopy(:,c.test{icv},:),[ttrial*Nte q]);
    Ttest = T(c.test{icv});
    [cv_acc(:,icv),~,softpreds,genplot{icv}] = standard_classifier_test(model,Xtest,Ytest,Ttest,options);
    LL(:,c.test{icv}) = reshape(sum(Ytest.*log(softpreds),2),[ttrial, Nte]);
    Ypreds(:,c.test{icv},:) = reshape(softpreds,[ttrial,Nte,q]);
    if verbose_CV
        fprintf(['CV iteration: ' num2str(icv),' of ',int2str(NCV),'\n']); 
    end
end
acc = mean(cv_acc(:));
cv_acc = mean(cv_acc,2);
LL = mean(LL,2);

% compute AUC:
for t=1:ttrial
    AUC_t = zeros(q);
    for i=1:q
        for j=(i+1):q
            % find valid samples:
            validtrials = union(find(Ycopy(t,:,i)),find(Ycopy(t,:,j)));
            ytemp = permute(Ycopy(t,validtrials,[i,j]),[2,3,1]);
            temp = exp(squeeze(Ypreds(t,validtrials,[i,j])) - max(squeeze(Ypreds(t,validtrials,[i,j])),[],2));
            temp = rdiv(temp,sum(temp,2));
            [temp,inds] = sort(temp(:,1),'descend');
            ytemp = ytemp(inds,:);
            for n=1:length(temp)
                TPr(n) = sum(ytemp(1:n,1))/sum(ytemp(:,1));
                p = temp(n,1);
                FPr(n) = sum(ytemp(1:n,2))/sum(ytemp(:,2));
            end
            AUC_t(i,j) = sum(diff([0,FPr,1,1]) .* [0,TPr,1]);
        end
    end
    AUC(t) = mean(AUC_t(logical(triu(ones(q),1))));
end



meanGenPlot = [];
if isfield(options,'generalisationplot') && options.generalisationplot
    for icv = 1:NCV
        meanGenPlot = cat(3,meanGenPlot,genplot{icv});
    end
    meanGenPlot = squeeze(mean(meanGenPlot,3));
else
end
end

function Y_out = multinomToBinary(Y_in)
    Y_out=zeros(length(Y_in),length(unique(Y_in)));
    for i=1:length(Y_in)
        Y_out(i,Y_in(i))=1;
    end
end