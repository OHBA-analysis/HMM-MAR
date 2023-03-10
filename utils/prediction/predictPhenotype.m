function [predictedY,predictedYD,YD,stats] = predictPhenotype (Yin,Din,options,varargin)
%
% Kernel ridge regression estimation using a distance matrix using (stratified) LOO. 
% Using this means that the HMM was run once, out of the cross-validation loop
%
% INPUT
% Yin       (no. subjects by 1) vector of phenotypic values to predict,
%           which can be continuous or binary. If a multiclass variable
%           is to be predicted, then Yin should be encoded by a
%           (no. subjects by no. classes) matrix, with zeros or ones
%           indicator entries.
% Din       (no. subjects by no. subjects) matrix calculated (for 
%           example) by hmm_kernel, computeDistMatrix or
%           computeDistMatrix_AVFC. For the Gaussian kernel, this should be
%           distances/divergences. For the linear kernel, this should be
%           the kernel itself (dot-product).   
% options   Struct with the prediction options, with fields:
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
%   + kernel - which kernel to use ('linear' or 'gaussian')
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
% predictedYD    predicted response,in the decounfounded space
% YD    response,in the decounfounded space
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
% adapted to different kernels:
% Christine Ahrends, Aarhus University, 2022

if ~isfield(options, 'kernel')
    kernel = 'linear';
else
    kernel = options.kernel;
end
if strcmp(kernel, 'gaussian')
    Din(eye(size(Din,1))==1) = 0; 
end
[N,q] = size(Yin);

which_nan = false(N,1); 
if q == 1
    which_nan = isnan(Yin);
    if any(which_nan)
        Yin = Yin(~which_nan);
        Din = Din(~which_nan,~which_nan);
        warning('NaN found on Yin, will remove...')
    end
    N = size(Yin,1);
end

if nargin < 3 || isempty(options), options = struct(); end
% if ~isfield(options,'method')
%     if isfield(options,'K')
%         method = 'K';
%     else
%         method = 'KRR';
%     end
% else
%     method = 'KRR';
% end
if ~isfield(options,'alpha')
    alpha = [0.0001 0.001 0.01 0.1 0.4 0.7 1.0 10 100];
else
    alpha = options.alpha;
end
if strcmp(kernel, 'gaussian')
    if ~isfield(options,'sigmafact')
        sigmafact = [1/5 1/3 1/2 1 2 3 5];
    else
        sigmafact = options.sigmafact;
    end
else
    sigmafact = 1;
end
if ~isfield(options,'K')
    KN = 1:min(50,round(0.5*N));
else
    KN = options.K; 
end

if ~isfield(options,'CVscheme'), CVscheme = [10 10];
else, CVscheme = options.CVscheme; end
if ~isfield(options,'CVfolds'), CVfolds = [];
else, CVfolds = options.CVfolds; end
% if ~isfield(options,'biascorrect'), biascorrect = 0;
% else, biascorrect = options.biascorrect; end
if ~isfield(options,'verbose'), verbose = 0;
else, verbose = options.verbose; end

% check correlation structure
allcs = []; 
if (nargin>3) && ~isempty(varargin{1})
    cs = varargin{1};
    if ~isempty(cs)
        is_cs_matrix = (size(cs,2) == size(cs,1));
        if is_cs_matrix 
            if any(which_nan)
                cs = cs(~which_nan,~which_nan);
            end
            [allcs(:,2),allcs(:,1)]=ind2sub([length(cs) length(cs)],find(cs>0));    
        else
            if any(which_nan)
                cs = cs(~which_nan);
            end
            allcs = find(cs > 0);
        end
    end
else, cs = []; 
end

% get confounds
if (nargin>4) && ~isempty(varargin{2})
    confounds = varargin{2};
    confounds = confounds - repmat(mean(confounds),N,1);
    deconfounding = 1;
    if any(which_nan)
        confounds = confounds(~which_nan,:);
    end
else
    confounds = []; deconfounding = 0;
end

Ymean = zeros(N,q);
YD = zeros(N,q); % deconfounded signal
YmeanD = zeros(N,q); % mean in deconfounded space
predictedY = zeros(N,q);
if deconfounding, predictedYD = zeros(N,q); end

% create the inner CV structure - stratified for family=multinomial
if isempty(CVfolds)
    if CVscheme(1)==1
        folds = {1:N};
    elseif q == 1
        
        Yin_copy = Yin; Yin_copy(isnan(Yin)) = realmax;
        folds = cvfolds(Yin_copy,CVscheme(1),allcs,1);
    else % no stratification
        
        folds = cvfolds(randn(size(Yin,1),1),CVscheme(1),allcs,1);
    end
else
    folds = CVfolds;
end

stats = struct();

for ifold = 1:length(folds)
    
    if verbose, fprintf('CV iteration %d \n',ifold); end
    
    J = folds{ifold}; % test
    if isempty(J), continue; end
    if length(folds)==1
        ji = J;
    else
        ji = setdiff(1:N,J); % train
    end
    
    D = Din(ji,ji); 
    Y = Yin(ji,:);
    D2 = Din(J,ji);
    
    % family structure for this fold
    Qallcs=[];
    if (~isempty(cs))
        if is_cs_matrix
            [Qallcs(:,2),Qallcs(:,1)] = ...
                ind2sub([length(cs(ji,ji)) length(cs(ji,ji))],find(cs(ji,ji)>0));
        else
            Qallcs = find(cs(ji) > 0);
        end
    end
        
    for ii = 1:q
        
        Dii = D; Yii = Y; % in case you use parfor
        
        ind = find(~isnan(Y(:,ii)));
        Yii = Yii(ind,ii); 
        QDin = Dii(ind,ind); 
        QN = length(ind);
        
        Qfolds = cvfolds(Yii,CVscheme(2),Qallcs,1); % we stratify

        % deconfounding business
        if deconfounding
            Cii = confounds(ji,:); Cii = Cii(ind,:);
            [betaY,interceptY,Yii] = deconfoundPhen(Yii,Cii);
        end
        
        % centering response
        my = mean(Yii); 
        Yii = Yii - repmat(my,size(Yii,1),1);
        Ymean(J,ii) = my;
        QYin = Yii;
        
        Dev = Inf(length(alpha),length(sigmafact));
        
        for isigm = 1:length(sigmafact)
            
            sigmf = sigmafact(isigm);
            
            QpredictedY = Inf(QN,length(alpha));
            QYinCOMPARE = QYin;
            
            % Inner CV loop
            for Qifold = 1:length(Qfolds)
                
                QJ = Qfolds{Qifold}; Qji=setdiff(1:QN,QJ);
                QD = QDin(Qji,Qji);
                QY = QYin(Qji,:); Qmy = mean(QY); QY = QY-Qmy;
                Nji = length(Qji);
                QD2 = QDin(QJ,Qji);
                
                if strcmp(kernel, 'gaussian')
                    sigmabase = auto_sigma(QD);
                    sigma = sigmf * sigmabase;
                    K = gauss_kernel(QD,sigma);
                    K2 = gauss_kernel(QD2,sigma);
                elseif strcmp(kernel, 'linear')
                    K = QD;
                    K2 = QD2;
                end
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
        alph = alpha(ialph);
        Dii = D(ind,ind); D2ii = D2(:,ind);
        
        if strcmp(kernel, 'gaussian')
            sigmf = sigmafact(isigm);
            sigmabase = auto_sigma(D);
            sigma = sigmf * sigmabase;
            K = gauss_kernel(Dii,sigma);
            K2 = gauss_kernel(D2ii,sigma);
        elseif strcmp(kernel, 'linear')
            K = Dii;
            K2 = D2ii;
            sigmf = NaN;
        end
        Nji = length(ind);
        I = eye(Nji);
        
        ridg_pen_scale = mean(diag(K));
        beta = (K + ridg_pen_scale * alph * I) \ Yii;
        
        % predict the test fold
        predictedY(J,ii) = K2 * beta + my; % some may be NaN actually
    
%     else % Nearest Neighbour estimator
%         Dev = Inf(length(K),1);
%         
%         QpredictedY = Inf(QN,length(K));
%         QYinCOMPARE = QYin;
%         
%         % Inner CV loop
%         for Qifold = 1:length(Qfolds)
%             
%             QJ = Qfolds{Qifold}; Qji=setdiff(1:QN,QJ);
%             QD = QDin(Qji,Qji);
%             QY = QYin(Qji,:); Qmy = mean(QY); QY=QY-Qmy;
%             
%             for j = 1:length(QJ)
%                 [~,order] = sort(QD(:,j));
%                 Yordered = QY(order,:);
%                 for ik = 1:length(K)
%                     k = K(ik);
%                     QpredictedY(QJ(j),ik) = mean(Yordered(1:k,:));
%                 end
%             end
%         end
%         
%         for ik = 1:length(K)
%             Dev(k) = (sum(( QpredictedY - ...
%                 repmat(QYinCOMPARE,1,length(K))).^2) / QN)';
%         end
%             
%         [~,ik] = min(Dev); % Pick the one with the lowest deviance
%         k = KN(ik);
%         for j = 1:length(J)
%             [~,order] = sort(D(:,j));
%             Yordered = Y(order,:);
%             predictedY(J(j)) = mean(Yordered(1:k,:));
%         end
%      
%     end
    
        % predictedYD and YD in deconfounded space; Yin and predictedY are confounded
        predictedYD(J,ii) = predictedY(J,ii);
        YD(J,ii) = Yin(J,ii);
        YmeanD(J,ii) = Ymean(J,ii);
        if deconfounding % in order to later estimate prediction accuracy in deconfounded space
            [~,~,YD(J,ii)] = deconfoundPhen(YD(J,ii),confounds(J,:),betaY,interceptY);
            % original space
            predictedY(J,ii) = confoundPhen(predictedY(J,ii),confounds(J,:),betaY,interceptY);
            Ymean(J,ii) = confoundPhen(YmeanD(J,ii),confounds(J,:),betaY,interceptY);
        end
    
%     if biascorrect % we do this in the original space
%         Yhattrain = K * beta + my;
%         if deconfounding
%             Yhattrain = confoundPhen(Yhattrain,confounds(ji,:),betaY,interceptY);
%             Ytrain = [confoundPhen(QYin,confounds(ji,:),betaY,interceptY) ...
%                 ones(size(QYin,1),1)];
%         else
%             Ytrain = [QYin ones(size(QYin,1),1)];
%         end
%         b = pinv(Ytrain) * Yhattrain;
%         predictedY(J,:) = (predictedY(J,:) - b(2)) / b(1);
%     end
    
    end
    
    %disp(['Fold ' num2str(ifold) ])
    stats.alpha(ifold) = alph;
    stats.sigma(ifold) = sigmf;


end

stats.sse = zeros(q,1);
stats.cod = zeros(q,1);
stats.corr = zeros(q,1);
stats.baseline_corr = zeros(q,1);
stats.pval = zeros(q,1);
if deconfounding
    stats.sse_deconf = zeros(q,1);
    stats.cod_deconf = zeros(q,1);
    stats.corr_deconf = zeros(q,1);
    stats.baseline_corr_deconf = zeros(q,1);
    stats.pval_deconf = zeros(q,1);
end

for ii = 1:q
    ind = find(~isnan(Yin(:,ii)));
    stats.sse(ii) = sum((Yin(ind,ii)-predictedY(ind,ii)).^2);
    nullsse = sum((Yin(ind,ii)-Ymean(ind,ii)).^2);
    stats.cod(ii) = 1 - stats.sse(ii) / nullsse;
    stats.corr(ii) = corr(Yin(ind,ii),predictedY(ind,ii));
    stats.baseline_corr(ii) = corr(Yin(ind,ii),Ymean(ind,ii));
    [~,pv] = corrcoef(Yin(ind,ii),predictedY(ind,ii)); % original space
    if corr(Yin(ind,ii),predictedY(ind,ii))<0, stats.pval(ii) = 1;
    else, stats.pval(ii) = pv(1,2);
    end
    if deconfounding
        stats.sse_deconf(ii) = sum((YD(ind,ii)-predictedYD(ind,ii)).^2);
        nullsse_deconf = sum((YD(ind,ii)-YmeanD(ind,ii)).^2);
        stats.cod_deconf(ii) = 1 - stats.sse_deconf(ii) / nullsse_deconf;
        stats.corr_deconf(ii) = corr(YD(ind,ii),predictedYD(ind,ii));
        stats.baseline_corr_deconf(ii) = corr(YD(ind,ii),YmeanD(ind,ii));
        [~,pv] = corrcoef(YD(ind,ii),predictedYD(ind,ii)); % original space
        if corr(YD(ind,ii),predictedYD(ind,ii))<0, stats.pval_deconf(ii) = 1;
        else, stats.pval_deconf(ii) = pv(1,2);
        end
    end
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
