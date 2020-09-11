function [acc_t,predictions_hard,predictions_soft,genplot] = standard_classifier_test(model,X,Y,T,options,binsize)
% Fit an already trained classifier given by the structure model to the 
% test data given in X with labels Y.
%
% INPUT
% model: a classification model outputted from the function
%           standard_classifier_train
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

if nargin < 5 || isempty(options), options = struct(); end
if nargin < 6, binsize = 1; end

if ~all(T==T(1)), error('All elements of T must be equal for time-aligned classification'); end 
N = length(T); ttrial = T(1); 

if size(Y,1) < size(Y,2); Y = Y'; end
q = size(Y,2);
if size(Y,1) == length(T) % one value per trial
    responses = Y;
elseif length(Y(:)) ~= (ttrial*N*q)
    error('Incorrect dimensions in Y')
else
    responses = reshape(Y,[ttrial N q]);
    Ystar = reshape(repmat(responses(1,:,:),[ttrial,1,1]),[ttrial*N q]);
    responses = permute(responses(1,:,:),[2 3 1]); % N x q
    if ~all(Y(:)==Ystar(:))
        error('For time-aligned classification, the same stimulus must be presented for the entire trial');
    end
end

Ycopy = Y;
if size(Ycopy,1) == N 
    Ycopy = repmat(reshape(Ycopy,[1 N q]),[ttrial 1 1]);
else
    Ycopy = reshape(Ycopy,[ttrial N q]);
end
% no demeaning by default if this is a classification problem
if ~isfield(options,'demeanstim'), options.demeanstim = 0; end

options.Nfeatures = 0; 
options.K = 1; 
%[X,Y,T] = preproc4hmm(X,Y,T,options); % this demeans Y
p = size(X,2);

%reshape to 3d vectors:
X = reshape(X,[ttrial N p]);
Y = reshape(Y,[ttrial N q]);

if (binsize/2)==0
    warning(['binsize must be an odd number, setting to ' num2str(binsize+1)])
    binsize = binsize + 1; 
end

if any(T~=ttrial), error('All trials must have the same length'); end
if nargin<5, binsize = 1; end

if ~isfield(model,'classifier')
    error('Model inputted does not conform to requirements; it should be trained using the function standard_classifier_train.m');
else
    classifier = model.classifier; 
    if ~(strcmp(classifier,'logistic') || strcmp(classifier,'SVM') ||...
            strcmp(classifier,'LDA') || strcmp(classifier,'KNN') || ...
            strcmp(classifier,'decisiontree') || strcmp(classifier,'SVM_rbf'));
        error('model.classifier can only take values "logistic", "SVM" or "LDA"');
    end
end
if strcmp(classifier,'logistic')
    Q_star = size(model.betas,3); %check if multinomial regression
else
    Q_star=q;
end
if ~isfield(options,'generalisationplot'),options.generalisationplot=false;end
if isfield(model,'lambda')
    L=length(model.lambda);
else
    L=1;
end

halfbin = floor(binsize/2); 
%nwin = round(ttrial / binsize);
%binsize = floor(ttrial / nwin); 
%RidgePen = lambda * eye(p);

if strcmp(classifier,'logistic')
    Y(Y==-1)=0;
    Y_pred = zeros(ttrial,N,Q_star,L);
    Y_pred_genplot=zeros(ttrial,ttrial,N,Q_star,L);
    for iLambda=1:L
        for t=1:ttrial
            if p>1
                Y_pred(t,:,:,iLambda)=squeeze(X(t,:,:))*permute(model.betas(t,:,:,iLambda),[2,3,1,4]) + repmat(model.intercepts(t,:,iLambda),N,1);
            else
                Y_pred(t,:,:,iLambda)=X(t,:)'*permute(model.betas(t,:,:,iLambda),[2,3,1,4]) + repmat(model.intercepts(t,:,iLambda),N,1);
            end
            if options.generalisationplot
                for t2=1:ttrial
                    Y_pred_genplot(t,t2,:,:,iLambda)= ...
                        squeeze(X(t2,:,:))*permute(model.betas(t,:,:,iLambda),[2,3,1,4]) + repmat(model.intercepts(t,:,iLambda),N,1);
                end
            end
        end
    end
    
    for iL=1:L
        if Q_star~=q
            pred_temp=multinomLogRegPred(Y_pred(:,:,:,iL));
            predictions_soft(:,:,iL)=reshape(pred_temp,[ttrial*N,q]);
            predictions_hard(:,:,iL)=hardmax(predictions_soft(:,:,iL));
        else
            Y_pred = reshape(Y_pred,[ttrial*N,q,L]);
            predictions_soft(:,:,iL)=log_sigmoid(Y_pred(:,:,iL));
            if Q_star>1
                predictions_hard(:,:,iL)=hardmax(Y_pred(:,:,iL));
            else
                predictions_hard(:,:,iL) = Y_pred(:,:,iL)>0;
            end
        end
    end
    if options.generalisationplot    
        if Q_star==q
            Y_pred_genplot = reshape(Y_pred_genplot,[ttrial, ttrial*N,Q_star,L]);
            for iL=1:iL
                for t=1:ttrial
                    Y_pred_genplot(t,:,:,iL) = hardmax(squeeze(Y_pred_genplot(t,:,:,iL)));
                end
            end
        else %multinomial:
            Y_pred_genplot_q = zeros([ttrial, ttrial,N,q,L]);
            for iL=1:iL;
                for t=1:ttrial
                    pred_temp = multinomLogRegPred(squeeze(Y_pred_genplot(t,:,:,:,iL)));
                    pred_temp = reshape(pred_temp,[ttrial*N,q]);
                    Y_pred_genplot_q(t,:,:,:,iL) = reshape(hardmax(pred_temp),[ttrial,N,q]);
                end
            end
            Y_pred_genplot=Y_pred_genplot_q;
        end
    end
elseif strcmp(classifier,'SVM') || strcmp(classifier,'SVM_rbf')
     % note this uses distance from hyperplane as soft score to allow
     % multiclass categorisation
     Y_pred = zeros(ttrial,N,q);
     Y_pred_genplot=zeros(ttrial,ttrial,N,q);
     for t=1:ttrial
        for iStim=1:q
            %[~,sc] = predict(model.SVM{t,iStim},squeeze(X(t,:,:)));
            [~,sc] = model.SVM{t,iStim}.predict(squeeze(X(t,:,:)));
            Y_pred(t,:,iStim)=sc(:,2);
            if options.generalisationplot  
                for t2=1:ttrial
                    %[~,sc] = predict(model.SVM{t,iStim},squeeze(X(t2,:,:)));
                    [~,sc] = model.SVM{t,iStim}.predict(squeeze(X(t2,:,:)));
                    Y_pred_genplot(t,t2,:,iStim)= sc(:,2);
                end
            end
        end
     end
     Y_pred = reshape(Y_pred,[ttrial*N,q,L]);
     predictions_soft=Y_pred;
     predictions_hard=hardmax(Y_pred);
     if options.generalisationplot    
        Y_pred_genplot = reshape(Y_pred_genplot,[ttrial, ttrial*N,q]);
        for t=1:ttrial
            Y_pred_genplot(t,:,:) = hardmax(squeeze(Y_pred_genplot(t,:,:)));
        end
    end
elseif strcmp(classifier,'LDA')
    X = reshape(X,[ttrial*N,p]);
    [predictions_hard, predictions_soft] = LDApredict(model,repmat(eye(ttrial),N,1),X);
    predictions_soft = exp(predictions_soft - repmat(max(predictions_soft,[],2),1,q));
    predictions_soft = rdiv(predictions_soft,sum(predictions_soft,2)); 
elseif strcmp(classifier,'KNN')
    if ~isfield(model,'K');model.K=1;end
    predictions_hard = zeros(ttrial,N,q);
    for t=1:ttrial
        candidates = squeeze(model.X_train(t,:,:));
        D=dist(squeeze(X(t,:,:)),candidates');
        for n=1:N
            [dists,inds]=sort(D(n,:));
            if options.K==1
                M=model.Y_train(t,inds(1:options.K),:);
                ind_winner = find(squeeze(M));
                predictions_hard(t,n,ind_winner) = 1;
            else
                Y_pred=squeeze(model.Y_train(t,inds(1:options.K),:));
                M = sum(Y_pred,1);
                [M,inds2] = sort(M,'descend');
                if M(1)>M(2)
                    predictions_hard(t,n,inds2(1)) = 1;
                else %tie break - just take closest point
                    ind_winner = find(squeeze(model.Y_train(t,inds(1),:)));
                    predictions_hard(t,n,inds(1)) = 1;
                end
            end
        end
    end
    predictions_hard = reshape(predictions_hard,[ttrial*N,q]);
    predictions_soft = predictions_hard;
elseif strcmp(classifier,'decisiontree');
    predictions_hard = zeros(ttrial,N,q);
    for t=1:ttrial
        temp = model.decisiontree{t}.predict(squeeze(X(t,:,:)));
        for n=1:N
            predictions_hard(t,n,temp(n))=1;
        end
    end
    predictions_hard = reshape(predictions_hard,[ttrial*N,q]);
    predictions_soft = predictions_hard;
end

%and compute accuracy metrics using hard classification output:
Y = reshape(logical(Y),[ttrial*N q]);
predictions_hard=logical(predictions_hard);
true_preds = all(~xor(predictions_hard,Y),2);
acc=mean(true_preds);
acc_t = mean(reshape(true_preds,[ttrial,N]),2);
if options.generalisationplot
    Y_true_genplot = repmat(permute(Ycopy,[4,1,2,3]),[ttrial,1,1,1]);
    Y_pred_genplot = reshape(Y_pred_genplot,[ttrial, ttrial,N,q]);
    accplot = sum(logical(Y_true_genplot) & logical(Y_pred_genplot),4);
    genplot = squeeze(mean(accplot,3));
else
    genplot=[];
end
end

function preds_hard = hardmax(Y_pred)
% assuming multiple binomial only; Y_pred of dimension [NT x q]
if ndims(Y_pred)>2
    error('Wrong dimensions entered for computing predictions');
end
[~,a] = max(Y_pred,[],2);
preds_hard = zeros(size(Y_pred));
for i=1:length(preds_hard)
    preds_hard(i,a(i))=1;
end

end