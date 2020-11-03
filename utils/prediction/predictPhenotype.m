function [predictedY,stats] = predictPhenotype (Yin,Din,options,varargin)
%
% Kernel ridge regression or nearest-neighbour estimation using
% a distance matrix using (stratified) LOO 
%
% INPUT
% Yin       (no. subjects by 1) vector of phenotypic values to predict,
%           which can be continuous or binary. If a multiclass variable
%           is to be predicted, then Yin should be encoded by a
%           (no. subjects by no. classes) matrix, with zeros or ones
%           indicator entries.
% Din       (no. subjects by no. subjects) matrix of distances between
%           subjects, calculated (for example) by computeDistMatrix or
%           computeDistMatrix_AVFC
% options   Struct with the prediction options, with fields:
%   + method - either 'KRR' for Kernel ridge regression or 'NN' for
%           Nearest-neighbour estimator
%   + alpha - for method='KRR', a vector of weights on the L2 penalty on the regression
%           By default: [0.0001 0.001 0.01 0.1 0.4 0.7 1.0 10 100]
%   + sigmafact - for method='KRR', a vector of parameters for the kernel; in particular,
%           this is the factor by which we will multiply a data-driven estimation
%           of the kernel parameter. By default: [1/5 1/3 1/2 1 2 3 5];
%   + K - for method='NN', a vector with the number of nearest neighbours to use
%   + CVscheme - vector of two elements: first is number of folds for model evaluation;
%             second is number of folds for the model selection phase (0 in both for LOO)
%   + CVfolds - prespecified CV folds for the outer loop
%   + biascorrect - whether we correct for bias in the estimation 
%                   (Smith et al. 2019, NeuroImage)
%   + verbose -  display progress?
% cs        optional (no. subjects X no. subjects) dependency structure matrix with
%           specifying possible relations between subjects (e.g., family
%           structure), or a (no. subjects X 1) vector defining some
%           grouping, with (1...no.groups) or 0 for no group
% confounds     (no. subjects by  no. of confounds) matrix of features that 
%               potentially influence the phenotypes and we wish to control for 
%               (optional)
%
% OUTPUT
% predictedY    predicted response,in the original (non-decounfounded) space
% stats         structure, with fields
%   + pval - permutation-based p-value, if permutation is run;
%            otherwise, correlation-based p-value
%   + cod - coeficient of determination 
%   + corr - correlation between predicted and observed Y 
%   + baseline_corr - baseline correlation between predicted and observed Y for null model 
%   + sse - sum of squared errors
%   + baseline_sse - baseline sum of squared errors
%   PLUS: All of the above +'_deconf' in the deconfounded space, if counfounds were specified
%   + alpha - selected values for alpha at each CV fold
%   + sigmaf - selected values for sigmafact at each CV fold
%
% Author: Diego Vidaurre, OHBA, University of Oxford
%         Steve Smith, fMRIB University of Oxford

Din(eye(size(Din,1))==1) = 0; 
N = size(Yin,1);
if nargin < 3 || isempty(options), options = struct(); end
if ~isfield(options,'method')
    if isfield(options,'K')
        method = 'K';
    else
        method = 'KRR';
    end
else
    method = 'KRR';
end
if ~isfield(options,'alpha')
    alpha = [0.0001 0.001 0.01 0.1 0.4 0.7 1.0 10 100];
else
    alpha = options.alpha;
end
if ~isfield(options,'sigmafact')
    sigmafact = [1/5 1/3 1/2 1 2 3 5];
else
    sigmafact = options.sigmafact;
end
if ~isfield(options,'K')
    K = 1:min(50,round(0.5*N));
else
    K = options.K; 
end

if ~isfield(options,'CVscheme'), CVscheme = [10 10];
else, CVscheme = options.CVscheme; end
if ~isfield(options,'CVfolds'), CVfolds = [];
else, CVfolds = options.CVfolds; end
if ~isfield(options,'biascorrect'), biascorrect = 0;
else, biascorrect = options.biascorrect; end
if ~isfield(options,'verbose'), verbose = 1;
else, verbose = options.verbose; end

% check correlation structure
allcs = []; 
if (nargin>4) && ~isempty(varargin{1})
    cs=varargin{1};
    if ~isempty(cs)
        is_cs_matrix = (size(cs,2) == size(cs,1));
        if is_cs_matrix 
            [allcs(:,2),allcs(:,1)]=ind2sub([length(cs) length(cs)],find(cs>0));    
            [grotMZi(:,2),grotMZi(:,1)]=ind2sub([length(cs) length(cs)],find(tril(cs,1)==1));
            [grotDZi(:,2),grotDZi(:,1)]=ind2sub([length(cs) length(cs)],find(tril(cs,1)==2));
        else
            allcs = find(cs > 0);
            grotMZi = find(cs == 1); 
            grotDZi = find(cs == 2); 
        end
    end
else, cs = []; 
end

% get confounds
if (nargin>5) && ~isempty(varargin{2})
    confounds = varargin{2};
    confounds = confounds - repmat(mean(confounds),N,1);
    deconfounding = 1;
else
    confounds = []; deconfounding = 0;
end

Ymean = zeros(size(Yin));
YD = zeros(size(Yin)); % deconfounded signal
YmeanD = zeros(size(Yin)); % mean in deconfounded space
predictedY = zeros(N,1);
if deconfounding, predictedYD = zeros(N,1); end

% create the inner CV structure - stratified for family=multinomial
if isempty(CVfolds)
    if CVscheme(1)==1
        folds = {1:N};
    else
        folds = cvfolds(Yin,CVscheme(1),allcs);
    end
else
    folds = CVfolds;
end

stats = struct();
stats.alpha = zeros(1,length(folds) );
stats.sigmaf = zeros(1,length(folds) );

for ifold = 1:length(folds)
    
    if verbose, fprintf('CV iteration %d \n',ifold); end
    
    J = folds{ifold}; % test
    if isempty(J), continue; end
    if length(folds)==1
        ji = J;
    else
        ji = setdiff(1:N,J); % train
    end
    QN = length(ji);
    D = Din(ji,ji); Y = Yin(ji,:);
    D2 = Din(J,ji);
    
    % family structure for this fold
    Qallcs=[];
    if (~isempty(cs))
        [Qallcs(:,2),Qallcs(:,1)] = ...
            ind2sub([length(cs(ji,ji)) length(cs(ji,ji))],find(cs(ji,ji)>0));
    end
    
    % deconfounding business
    if deconfounding
        [betaY,interceptY,Y] = deconfoundPhen(Y,confounds(ji,:));
    end
    
    % centering response
    my = mean(Y); Y = Y - my;
    YmeanD(J) = my;
    
    QDin = Din(ji,ji);
    QYin = Y;
    Qfolds = cvfolds(Y,CVscheme(2),Qallcs);
 
    if strcmp(method,'KRR') % Kernel regression
        Dev = Inf(length(alpha),length(sigmafact));
        
        for isigm = 1:length(sigmafact)
            
            sigmf = sigmafact(isigm);
            
            QpredictedY = Inf(QN,length(alpha));
            QYinCOMPARE = QYin;
            
            % Inner CV loop
            for Qifold = 1:length(Qfolds)
                
                QJ = Qfolds{Qifold}; Qji=setdiff(1:QN,QJ);
                QD = QDin(Qji,Qji);
                QY = QYin(Qji,:); Qmy = mean(QY); QY=QY-Qmy;
                Nji = length(Qji);
                QD2 = QDin(QJ,Qji);
                
                sigmabase = auto_sigma(QD);
                sigma = sigmf * sigmabase;
                
                K = gauss_kernel(QD,sigma);
                K2 = gauss_kernel(QD2,sigma);
                I = eye(Nji);
                ridg_pen_scale = mean(diag(K));
                
                for ialph = 1:length(alpha)
                    alph = alpha(ialph);
                    beta = (K + ridg_pen_scale * alph * I) \ QY;
                    QpredictedY(QJ,ialph) = K2 * beta + repmat(Qmy,length(QJ),1);
                end
            end
            
            Dev(:,isigm) = (sum(( QpredictedY - ...
                repmat(QYinCOMPARE,1,length(alpha))).^2) / QN)';
            
        end
        
        [~,m] = min(Dev(:)); % Pick the one with the lowest deviance
        [ialph,isigm] = ind2sub(size(Dev),m);
        sigmf = sigmafact(isigm);
        sigmabase = auto_sigma(D);
        sigma = sigmf * sigmabase;
        alph = alpha(ialph);
        
        K = gauss_kernel(D,sigma);
        K2 = gauss_kernel(D2,sigma);
        Nji = length(ji);
        I = eye(Nji);
        
        ridg_pen_scale = mean(diag(K));
        beta = (K + ridg_pen_scale * alph * I) \ Y; 
        
        % predict the test fold
        predictedY(J) = K2 * beta + my;
    
    else % Nearest Neighbour estimator
        Dev = Inf(length(K),1);
        
        QpredictedY = Inf(QN,length(K));
        QYinCOMPARE = QYin;
        
        % Inner CV loop
        for Qifold = 1:length(Qfolds)
            
            QJ = Qfolds{Qifold}; Qji=setdiff(1:QN,QJ);
            QD = QDin(Qji,Qji);
            QY = QYin(Qji,:); Qmy = mean(QY); QY=QY-Qmy;
            
            for j = 1:length(QJ)
                [~,order] = sort(QD(:,j));
                Yordered = QY(order,:);
                for ik = 1:length(K)
                    k = K(ik);
                    QpredictedY(QJ(j),ik) = mean(Yordered(1:k,:));
                end
            end
        end
        
        for ik = 1:length(K)
            Dev(k) = (sum(( QpredictedY - ...
                repmat(QYinCOMPARE,1,length(K))).^2) / QN)';
        end
            
        [~,ik] = min(Dev); % Pick the one with the lowest deviance
        k = K(ik);
        for j = 1:length(J)
            [~,order] = sort(D(:,j));
            Yordered = Y(order,:);
            predictedY(J(j)) = mean(Yordered(1:k,:));
        end
     
    end
    
    % predictedYD and YD in deconfounded space; Yin and predictedY are confounded
    predictedYD(J,:) = predictedY(J,:);
    YD(J,:) = Yin(J,:);
    YmeanD(J) = Ymean(J,:);
    if deconfounding % in order to later estimate prediction accuracy in deconfounded space
        [~,~,YD(J,:)] = deconfoundPhen(YD(J,:),confounds(J,:),betaY,interceptY);
        % original space
        predictedY(J,:) = confoundPhen(predictedY(J,:),confounds(J,:),betaY,interceptY); 
        Ymean(J) = confoundPhen(YmeanD(J,:),confounds(J,:),betaY,interceptY);
    end
    
    if biascorrect % we do this in the original space
        Yhattrain = K * beta + my;
        if deconfounding
            Yhattrain = confoundPhen(Yhattrain,confounds(ji,:),betaY,interceptY);
            Ytrain = [confoundPhen(QYin,confounds(ji,:),betaY,interceptY) ...
                ones(size(QYin,1),1)];
        else
            Ytrain = [QYin ones(size(QYin,1),1)];
        end
        b = pinv(Ytrain) * Yhattrain;
        predictedY(J,:) = (predictedY(J,:) - b(2)) / b(1);
    end
    
    stats.alpha(ifold) = alph;
    stats.sigmaf(ifold) = sigmf;
    
end

stats.sse = sum((Yin-predictedY).^2);
nullsse = sum((Yin-Ymean).^2);
stats.cod = 1 - stats.sse / nullsse;
stats.corr = corr(Yin,predictedY);
stats.baseline_corr = corr(Yin,Ymean);
[~,pv] = corrcoef(Yin,predictedY); % original space
if corr(Yin,predictedY)<0, stats.pval = 1; 
else, stats.pval=pv(1,2);
end

stats.sse_deconf = sum((YD-predictedYD).^2);
nullsse_deconf = sum((YD-YmeanD).^2);
stats.cod_deconf = 1 - stats.sse_deconf / nullsse_deconf;
stats.corr_deconf = corr(YD,predictedYD);
stats.baseline_corr_deconf = corr(YD,YmeanD);
[~,pv] = corrcoef(YD,predictedYD); % original space
if corr(YD,predictedYD)<0, stats.pval_deconf = 1; 
else, stats.pval_deconf=pv(1,2);
end

end


function K = gauss_kernel(D,sigma)
% Gaussian kernel
D = D.^2; % because distance is sqrt-ed
K = exp(-D/(2*sigma^2));
end


function sigma = auto_sigma (D)
% gets a data-driven estimation of the kernel parameter
D = D(triu(true(size(D,1)),1));
sigma = median(D);
end


function [betaY,my,Y] = deconfoundPhen(Y,confX,betaY,my)
if nargin<3, betaY = []; end
if isempty(betaY)
    my = mean(Y);
    Y = Y - my;
    betaY = (confX' * confX + 0.00001 * eye(size(confX,2))) \ confX' * Y;
end
res = Y - confX*betaY;
Y = res;
end


function Y = confoundPhen(Y,conf,betaY,my) 
Y = Y+conf*betaY+my;
end