% krr_predict_FK - kernel ridge regression estimation 
% using (stratified) LOO and permutation testing
% Diego Vidaurre 
% adapted to use feature matrix as input, different kernels (Fisher kernel)
% Christine Ahrends
%
% [predictedY,stats] = krr_predict_FK(Yin,Xin,family,parameters);
% [predictedY,stats] = krr_predict_FK(Yin,Xin,family,parameters,correlation_structure);
% [predictedY,stats] = krr_predict_FK(Yin,Xin,family,parameters,correlation_structure,Permutations);
% [predictedY,stats] = krr_predict_FK(Yin,Xin,family,parameters,correlation_structure,Permutations,confounds);
%
% INPUTS
% Yin - response vector (samples X 1)
% Xin - predictor matrix (samples X features)
% family - probability distribution of the response, one of the following:
%   + 'gaussian': standard linear regression on a continuous response
%   + 'poisson': non-negative counts
%   + 'multinomial': a binary-valued matrix with as columns as classes
%   + 'cox': a two-column matrix with the 1st column for time and the 2d for status: 1 for death and 0 right censored.
%   If a list with two elements, the first is the family for the feature selection stage (typically gaussian)
% parameters is a structure with:
%   + alpha - a vector of weights on the L2 penalty on the regression
%           coefficients, if 'Method' is 'lasso' or 'glmnet'; otherwise, it is the
%           values of the ridge penalty. 
%   + deconfounding - Which deconfounding strategy to follow if confounds 
%           are specified: if 0, no deconfounding is applied; if 1, confounds
%           are regressed out; if 2, confounds are regressed out using cross-validation
%   + CVscheme - vector of two elements: first is number of folds for model evaluation;
%             second is number of folds for the model selection phase (0 in both for LOO)
%   + CVfolds - prespecified CV folds for the outer loop
%   + Nperm - number of permutations (set to 0 to skip permutation testing)
%   + show_scatter - set to 1 to show a scatter plot of predicted_Y vs Y (only for family='gaussian' or 'poisson')
%   + verbose -  display progress?
%   + kernel - linear or Gaussian
% correlation_structure (optional) - A (Nsamples X Nsamples) matrix with
%                                   integer dependency labels (e.g., family structure), or 
%                                    A (Nsamples X 1) vector defining some
%                                    grouping: (1...no.groups) or 0 for no group
% Permutations (optional but must also have correlation_structure) - pre-created set of permutations
% confounds (optional) - features that potentially influence the inputs, and the outputs for family="gaussian'
%
% OUTPUTS
% predictedY - predicted response,in the original (non-decounfounded) space
% stats structure, with fields
%   + pval - permutation-based p-value, if permutation is run;
%            otherwise, correlation-based (family=gaussian) or multinomial-based p-value (family='multinomial')
%   + dev - cross-validated deviance (for family='gaussian', this is the sum of squared errors)
%   + cod - coeficient of determination
%   + dev_deconf - cross-validated deviance in the deconfounded space (family='gaussian')
%   + cod_deconf - coeficient of determination in the deconfounded space (family='gaussian')
%   + accuracy - cross-validated classification accuracy (family='multinomial')
% predictedYC - the predicted response, in the deconfounded space
% YoutC - the actual response, in the deconfounded space
% predictedYmean - the estimated mean of the response, in original space
% beta - the regression coefficients, which correspond to standarized predictors
% Cin - the mean covariance matrix, used for the riemannian transformation
% grotperms - the deviance values obtained for each permutation


function [predictedY,stats,predictedYC,beta] = krr_predict_FK(Yin,Xin,parameters,varargin)

N = size(Yin,1);
if isempty(parameters), parameters = struct(); end
if nargin<3, parameters = struct(); end
if ~isfield(parameters,'alpha')
    alpha = [0.00001 0.0001 0.001 0.01 0.1 0.4 0.7 1.0 10];
else
    alpha = parameters.alpha;
end
if ~isfield(parameters,'sigmafact')
    sigmafact = [1/5 1/3 1/2 1 2 3 5];
else
    sigmafact = parameters.sigmafact;
end

if ~isfield(parameters,'CVscheme'), CVscheme = [10 10];
else, CVscheme = parameters.CVscheme; end
if ~isfield(parameters,'CVfolds'), CVfolds = [];
else, CVfolds = parameters.CVfolds; end
if ~isfield(parameters,'deconfounding'), deconfounding=1;
else, deconfounding = parameters.deconfounding; end
if ~isfield(parameters,'biascorrect'), biascorrect = 0;
else, biascorrect = parameters.biascorrect; end
if ~isfield(parameters,'Nperm'), Nperm=1;
else, Nperm = parameters.Nperm; end
if ~isfield(parameters,'verbose'), verbose=0;
else, verbose = parameters.verbose; end
if ~isfield(parameters,'kernel'); kernel = 'gaussian';
else, kernel = parameters.kernel; end
if strcmpi(kernel, 'linear')
    sigmafact = 1;
end

if deconfounding==1
    parameters_dec = struct();
    parameters_dec.alpha = [0.0001 0.001 0.01 0.1 1 10 100 1000];
    parameters_dec.CVscheme = [10 10];
end
if deconfounding==2
   error('This deconfounding strategy is not yet available , must be 0 or 1') 
end

tmpnm = tempname; mkdir(tmpnm); mkdir(strcat(tmpnm,'/out')); mkdir(strcat(tmpnm,'/params')); 

    
Yin2 = Yin; % used for the variable selection stage

rng('shuffle')
if (Nperm<2),  Nperm=1;  end
if (nargin>3) && ~isempty(varargin{1})
    cs=varargin{1};
    if ~isempty(cs)
        if size(cs,2)>1 % matrix format
            [allcs(:,2),allcs(:,1)]=ind2sub([length(cs) length(cs)],find(cs>0));    
            [grotMZi(:,2),grotMZi(:,1)]=ind2sub([length(cs) length(cs)],find(tril(cs,1)==1));
            [grotDZi(:,2),grotDZi(:,1)]=ind2sub([length(cs) length(cs)],find(tril(cs,1)==2));
        else
            allcs = [];
            nz = cs>0; 
            gr = unique(cs(nz));  
            for g=gr'
               ss = find(cs==g);
               for s1=ss
                   for s2=ss
                       allcs = [allcs; [s1 s2]];
                   end
               end
            end
            % grotMZi and grotDZi need to be computer here
        end
    end
else
    cs = [];
end
if ~exist('allcs','var'), allcs = []; end
PrePerms=0;
if (nargin>4) && ~isempty(varargin{2})
    Permutations=varargin{2};
    if ~isempty(Permutations)
        PrePerms=1;
        Nperm=size(Permutations,2);
    end
else
    Permutations = [];
end
% get confounds, and deconfound Xin
if (nargin>5) && ~isempty(varargin{3})
    confounds = varargin{3};
    confounds = confounds - repmat(mean(confounds),N,1);
else
    confounds = []; deconfounding = 0;
end



YinORIG = Yin; 
YinORIGmean = zeros(size(Yin));
YinORIG2 = Yin2;
%if exist('Yin2','var'), YinORIG2=Yin2; YinORIGmean2 = zeros(size(Yin2)); end 
grotperms = zeros(Nperm,1);
YC = zeros(size(Yin)); % deconfounded signal
YCmean = zeros(size(Yin)); % mean in deconfounded space

for perm = 1:Nperm
    if (perm>1)
        if isempty(cs)           % simple full permutation with no correlation structure
            rng('shuffle');
            rperm = randperm(N);
            Yin=YinORIG(rperm,:);
            Yin2=YinORIG2(rperm,:);
        elseif (PrePerms==0)          % complex permutation, taking into account correlation structure
            PERM=zeros(1,N);
            rng('shuffle');
            perm1=randperm(size(grotMZi,1));
            for ipe=1:length(perm1)
                if rand<0.5, wt=[1 2]; else wt=[2 1]; end
                PERM(grotMZi(ipe,1))=grotMZi(perm1(ipe),wt(1));
                PERM(grotMZi(ipe,2))=grotMZi(perm1(ipe),wt(2));
            end
            rng('shuffle');
            perm1=randperm(size(grotDZi,1));
            for ipe=1:length(perm1)
                if rand<0.5, wt=[1 2]; else wt=[2 1]; end
                PERM(grotDZi(ipe,1))=grotDZi(perm1(ipe),wt(1));
                PERM(grotDZi(ipe,2))=grotDZi(perm1(ipe),wt(2));
            end
            rng('shuffle');
            from=find(PERM==0);  pto=randperm(length(from));  to=from(pto);  PERM(from)=to;
            Yin=YinORIG(PERM,:);
            Yin2=YinORIG2(PERM,:); 
        else                   % pre-supplied permutation
            Yin=YinORIG(Permutations(:,perm),:);  % or maybe it should be the other way round.....?
            Yin2=YinORIG2(Permutations(:,perm),:); 
        end
    end
    
    predictedYp = zeros(N,1);
    if deconfounding, predictedYpC = zeros(N,1); end
    
    % create the inner CV structure - stratified for family=multinomial
    if isempty(CVfolds)
        if CVscheme(1)==1
            folds = {1:N};
        else            
            rng('shuffle');
            folds = cvfolds_FK(Yin,'gaussian',CVscheme(1),allcs);
        end
    else
        folds = CVfolds;
    end
    
    if perm==1
        stats = struct();
        stats.alpha = zeros(1,length(folds) );
        stats.sigma = zeros(1,length(folds) );
    end
    
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
%         D = Din(ji,ji); Y = Yin(ji,:); 
%         D2 = Din(J,ji);
        X = Xin(ji,:); Y = Yin(ji,:);
        X2 = Xin(J,:);
        
        % family structure for this fold
        Qallcs=[]; 
        if (~isempty(cs))
            [Qallcs(:,2),Qallcs(:,1)]=ind2sub([length(cs(ji,ji)) length(cs(ji,ji))],find(cs(ji,ji)>0));  
        end
                
        % deconfounding business
        if deconfounding==1
            [~,~,~,betaY,interceptY,Y] = nets_deconfound([],Y,confounds(ji,:),'gaussian',[],[],[],[],tmpnm);
        end
        
        % centering response
        my = mean(Y); Y = Y - my;
        YCmean(J) = my;
        
%         QDin = Din(ji,ji);
        QXin = Xin(ji,:);
                
        % uncomment this if you want to deconfound in inner loop
        %QYin = Yin(ji,:); 
        QYin = Y;
        rng('shuffle');
        Qfolds = cvfolds_FK(Y,'gaussian',CVscheme(2),Qallcs);
        
        % Variable selection with the elastic net
        Dev = Inf(length(alpha),length(sigmafact));
        
        for isigm = 1:length(sigmafact)
            
            sigmf = sigmafact(isigm);
            
            QpredictedYp = Inf(QN,length(alpha));
            QYinCOMPARE=QYin;
            
            % Inner CV loop
            for Qifold = 1:length(Qfolds)
                
                QJ = Qfolds{Qifold}; Qji=setdiff(1:QN,QJ); 
                
%                 QD = QDin(Qji,Qji); 
                QX = QXin(Qji,:);
                
                QY = QYin(Qji,:); Qmy = mean(QY); QY=QY-Qmy;
                Nji = length(Qji);
                
%                QD2 = QDin(QJ,Qji);
                QX2 = QXin(QJ,:);
                            
%                 K = gauss_kernel(QD,sigma); 
%                 K2 = gauss_kernel(QD2,sigma); 
                if strcmpi(kernel, 'gaussian')
                    sigmabase = auto_sigma2(QX);
                    sigma = sigmf * sigmabase;
                    K = gauss_kernel2(QX,QX,sigma);
                    K2 = gauss_kernel2(QX2,QX,sigma);
                elseif strcmpi(kernel, 'linear')
                    K = fisherK(QX,QX);
                    K2 = fisherK(QX2,QX);
                end
                
                I = eye(Nji);
                ridg_pen_scale = mean(diag(K));
                
                for ialph = 1:length(alpha)
                    
                    alph = alpha(ialph);
                    beta = (K + ridg_pen_scale * alph * I) \ QY;
                    
                    QpredictedYp(QJ,ialph) = K2 * beta + repmat(Qmy,length(QJ),1);
                end
            end
            
            % Pick the one with the lowest deviance
            Dev(:,isigm) = (sum(( QpredictedYp - ...
                repmat(QYinCOMPARE,1,length(alpha))).^2) / QN)'; 
            
        end
        
        [~,m] = min(Dev(:));
        [ialph,isigm] = ind2sub(size(Dev),m);
        alph = alpha(ialph); 
         
%         K = gauss_kernel(D,sigma); 
%         K2 = gauss_kernel(D2,sigma);
        if strcmpi(kernel, 'gaussian')
            sigmf = sigmafact(isigm);
            sigmabase = auto_sigma2(X);
            sigma = sigmf * sigmabase;
            K = gauss_kernel2(X,X,sigma);
            K2 = gauss_kernel2(X2,X,sigma);
        elseif strcmpi(kernel, 'linear')
            K = fisherK(X,X);
            K2 = fisherK(X2,X);
            sigmf = NaN;
        end
        Nji = length(ji); 
        I = eye(Nji);
        
        ridg_pen_scale = mean(diag(K));
        beta = (K + ridg_pen_scale * alph * I) \ Y;

        % predict the test fold
        predictedYp(J) = K2 * beta + repmat(my,length(J),1);
           
        % predictedYpC and YC in deconfounded space; Yin and predictedYp are confounded
        predictedYpC(J,:) = predictedYp(J,:); 
        YC(J,:) = Yin(J,:);
        YinORIGmean(J) = YCmean(J,:);
        if deconfounding % in order to later estimate prediction accuracy in deconfounded space
            [~,~,~,~,~,YC(J,:)] = nets_deconfound([],YC(J,:),confounds(J,:),'gaussian',[],[],betaY,interceptY,tmpnm);
            if ~isempty(betaY)
                predictedYp(J,:) = nets_confound(predictedYp(J,:),confounds(J,:),'gaussian',betaY,interceptY); % original space
                YinORIGmean(J) = nets_confound(YCmean(J,:),confounds(J,:),'gaussian',betaY,interceptY); 
            end
        end
        
        if biascorrect % we do this in the original space
            Yhattrain = Xin(ji,groti) * beta_final + repmat(my,length(ji),1);
            if deconfounding(2)
                Yhattrain = nets_confound(Yhattrain,confounds(ji,:),'gaussian',betaY,interceptY);
            end
            Ytrain = [QYin ones(size(QYin,1),1)]; 
            b = pinv(Ytrain) * Yhattrain;
            predictedYp(J,:) = (predictedYp(J,:) - b(2)) / b(1);
        end
        
        if perm==1
            stats.alpha(ifold) = alph;
            stats.sigma(ifold) = sigmf;
        end
        
    end
    
    % grotperms computed in deconfounded space
    grotperms(perm) = sum((YC-predictedYpC).^2);
      
    if perm==1
        predictedY = predictedYp;
        predictedYmean = YinORIGmean; 
        predictedYC = predictedYpC;
        YoutC = YC;
        stats.dev = sum((YinORIG-predictedYp).^2);
        stats.nulldev = sum((YinORIG-YinORIGmean).^2);
        stats.corr = corr(YinORIG,predictedYp);
        stats.cod = 1 - stats.dev / stats.nulldev;
        if Nperm==1
            [~,pv] = corrcoef(YinORIG,predictedYp); stats.pval=pv(1,2);
            if corr(YinORIG,predictedYp)<0, pv = 1; end
        end
    else
        fprintf('Permutation %d \n',perm)
    end
end

if Nperm>1 
    stats.pval = sum(grotperms<=grotperms(1)) / (Nperm+1);
end

system(['rm -fr ',tmpnm]);

end

function K = gauss_kernel(D,sigma)
% Gaussian kernel
D = D.^2; % because distance is sqrt-ed
K = exp(-D/(2*sigma^2));
end    

function K = gauss_kernel2(X,X2,sigma) 
for i = 1:size(X,1)
    for j = 1:size(X2,1)
        X3(i,j) = sqrt(sum(abs(X(i,:)-X2(j,:)).^2)).^2;
        %disp(['computing the norm now for i=' num2str(i) ' and j=' num2str(j) ' to get kernel']);
    end
end
K = exp(-X3/(2*sigma^2)); 
end
    

function sigma = auto_sigma2(X)
% gets a data-driven estimation of the kernel parameter
for i = 1:size(X,1)
    for j = 1:size(X,1)
        X3(i,j) = sqrt(sum(abs(X(i,:)-X(j,:)).^2));
        %disp(['computing the norm now for i=' num2str(i) ' and j=' num2str(j) ' to get sigma']);
    end
end
X4 = X3(triu(true(size(X3,1)),1));
sigma = median(X4);
end

function sigma = auto_sigma(D)
% gets a data-driven estimation of the kernel parameter
D = D(triu(true(size(D,1)),1));
sigma = median(D);
end

function K = fisherK(X,X2)
K = X*X2'; 
end

function K = fisherK_poly2(D,D2)
K = D*D2';
K = (1+K).^2;
end

function K = fisherK_poly3(X,X2)
K = X*X2';
K = (1+K).^3;
end