function [cv_acc,acc,genplot] = standard_classification(X,Y,T,options,binsize)
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
%           classifier      String indicating calssifer type;
%                   logistic    Logistic Regression classifier (default)
%                   SVM         Linear support vector machine classifier
%                   LDA         Linear Discriminant Analysis classifier
%           regularisation  String indicating type of regularisation;
%                   L1          Lasso regularisation
%                   L2          Ridge regularisation (default)
%                   ARD         Automatic relevance determination
%           lambda          Vector of regularisation penalty values to be
%                           tested
% binsize: NOT YET IMPLMENTED 
%           how many consecutive time points will be used for each estiamtion. 
%           By default, this is 1, such that one decoding model
%               is estimated using 1 time point
%
% OUTPUT
% 
%
% Author: Cam Higgins, OHBA, University of Oxford (2019)

if ~isfield(options,'classifier')
    options.classifier='logistic'; %set default
end
classifier=options.classifier;
do_preproc = 1; 

N = length(T); q = size(Y,2); ttrial = T(1); p = size(X,2); 
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
if ~isfield(options,'verbose'),options.verbose=0;end

% Preproc data and put in the right format 
if do_preproc
    if isfield(options,'embeddedlags'), el = options.embeddedlags;else;el=0;end
    options.K=0;
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
Ycopy = reshape(Ycopy,[ttrial N q]);

% Fit classifier to each fold:
acc=zeros(NCV);cv_acc=zeros(ttrial,NCV);
for icv = 1:NCV
    % train classifier on training set:
    Ntr = length(c.training{icv}); Nte = length(c.test{icv});
    Xtrain = reshape(X(:,c.training{icv},:),[Ntr*ttrial p] ) ;
    Ytrain = reshape(Y(:,c.training{icv},:),[Ntr*ttrial Q_star] ) ;
    Ttr = T(c.training{icv});
    model = standard_classifier_train(Xtrain,Ytrain,Ttr,options);
    
    % and test on test set:
    Nte = length(c.test{icv});
    Xtest = reshape(X(:,c.test{icv},:),[ttrial*Nte p]);
    Ytest = reshape(Ycopy(:,c.test{icv},:),[ttrial*Nte q]);
    Ttest = T(c.test{icv});
    [cv_acc(:,icv),~,~,genplot{icv}] = standard_classifier_test(model,Xtest,Ytest,Ttest,options);
    
    fprintf(['CV iteration: ' num2str(icv),' of ',int2str(NCV),'\n']); 
    
end
acc=mean(cv_acc(:));
cv_acc=mean(cv_acc,2);
end

function Y_out = multinomToBinary(Y_in)
    Y_out=zeros(length(Y_in),length(unique(Y_in)));
    for i=1:length(Y_in)
        Y_out(i,Y_in(i))=1;
    end
end