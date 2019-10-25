function model = standard_classifier_train(X,Y,T,options,binsize)
% Fit a classifier to the data given in X with labels Y of the type
% specified in the options - see below for details
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

if nargin < 4 || isempty(options), options = struct(); end
if nargin < 5, binsize = 1; end

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
[X,Y,T] = preproc4hmm(X,Y,T,options); % this demeans Y
Q_star=size(Y,2); % for multinomial and LDA (mean inserted) models

p = size(X,2);

X = reshape(X,[ttrial N p]);
Y = reshape(Y,[ttrial N Q_star]);

if (binsize/2)==0
    warning(['binsize must be an odd number, setting to ' num2str(binsize+1)])
    binsize = binsize + 1; 
end

if any(T~=ttrial), error('All trials must have the same length'); end
if nargin<5, binsize = 1; end

if isfield(options,'classifier')
    classifier = options.classifier; options = rmfield(options,'classifier');
    if ~(strcmp(classifier,'logistic') || strcmp(classifier,'SVM') || strcmp(classifier,'LDA'));
        error('options.classifier can only take values "logistic", "SVM" or "LDA"');
    end
else, classifier = 'logistic'; 
end
model=struct();
model.classifier = classifier;
if isfield(options,'lambda') 
    lambda = options.lambda; options = rmfield(options,'lambda');
else, lambda = 0.001; 
end
if isfield(options,'regularisation') 
    regtype = options.regularisation; options = rmfield(options,'regularisation');
    if ~(strcmp(regtype,'L1') || strcmp(regtype,'L2') || strcmp(regtype,'ARD'));
        error('options.regularisation can only take values "L1", "L2" or "ARD"');
    end
else, regtype = 'L2'; 
end


halfbin = floor(binsize/2); 
%nwin = round(ttrial / binsize);
%binsize = floor(ttrial / nwin); 
%RidgePen = lambda * eye(p);

%reshape to 3d vectors:
if strcmp(classifier,'logistic')
    % alternative: model = synchronousLogReg(X,Y,T,options)
    if length(unique(Ystar))<=2
        Y(Y==-1)=0;
    end
    for t=1:ttrial
        if options.verbose
            if strcmp(regtype,'L1')
                fprintf(['\nRunning LASSO Regression analysis on data for t=',int2str(t),'th sample']);
            elseif strcmp(regtype,'L2')
                fprintf(['\nRunning RIDGE Regression analysis on data for t=',int2str(t),'th sample']);
            elseif strcmp(regtype,'ARD')
                error('ARD regularisation not implemented yet for logistic regression');
            end
        end
        for ilambda=1:length(lambda)
            betas=zeros(p,Q_star);
            intercepts=zeros(1,Q_star);
            for c=1:q
                if length(unique(Ystar))>2
                    vp=Y(t,:,c)~=0;
                    Ysample = Y(t,vp,c)'==1;
                else
                    vp=true(1,N);
                    Ysample = Y(t,vp,c)';
                end
                if strcmp(regtype,'L1')
                    [betas(:,c),fitInfo]=lassoglm(squeeze(X(t,vp,:)),Ysample, ...
                        'binomial','Alpha', 1, 'Lambda', lambda(ilambda),'Standardize', false);
                    intercepts(c) = fitInfo.Intercept;
                elseif strcmp(regtype,'L2')
                     [betas(:,c),fitInfo]=lassoglm(squeeze(X(t,vp,:)),Ysample, ...
                            'binomial','Alpha', 0.00001, 'Lambda', lambda(ilambda),'Standardize', false);
                    intercepts(c) = fitInfo.Intercept;
                end
            end
            model.betas(t,:,:,ilambda)=betas;
            model.intercepts(t,:,ilambda)=intercepts;
        end
        model.lambda=lambda;    
    end
elseif strcmp(classifier,'SVM')
    for t=1:ttrial
        if options.verbose;
            fprintf(['Inferring SVM classifiers for t = ',int2str(t),'th sample\n']);
        end
        for iStim=1:Q_star
            model.SVM{t,iStim} = fitcsvm(squeeze(X(t,:,:)),squeeze(Y(t,:,iStim)));%,ones(sum(tselect(:,t))));
        end
    end
elseif strcmp(classifier,'LDA')
    options_LDA=struct;
    options_LDA.useUnsupervisedGamma=1; %this ensures gamma is not inferred
    options_LDA.covtype = 'diag';
    options_LDA.verbose = 1;
    options_LDA.useParallel = 0;
    options_LDA.standardise = 0;
    options_LDA.K=ttrial;
    options_LDA.Gamma=repmat(eye(ttrial),length(T),1);
    X_LDA = reshape(X,[ttrial*N p]);
    Y_LDA = reshape(Y,[ttrial*N Q_star]);
    if options.verbose
        fprintf(['Inferring LDA classifiers for all timepoints \n']);
    end
    model = encodertrain(X_LDA,Y_LDA,T,options_LDA);
    model.Gamma = model.Gamma(1:ttrial,:);
    model=rmfield(model,{'train','prior','Dir_alpha','Dir2d_alpha','P','Pi','features','K'});
    model.classifier='LDA';
end


end