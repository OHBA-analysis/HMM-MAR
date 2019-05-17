function [acc,acc_star,Ypred,Ypred_star,c] = tudacv(X,Y,T,options,do_preproc)

if nargin < 5, do_preproc = 1; end
N = length(T); q = size(Y,2); ttrial = T(1); p = size(X,2); K = options.K;
if ~all(T==T(1)), error('All elements of T must be equal for cross validation'); end 

if size(Y,1) == length(T) % one value per trial
    responses = Y;
    Ystar = reshape(repmat(reshape(responses,[1 N q]),[ttrial,1,1]),[ttrial*N q]);
else
    responses = reshape(Y,[ttrial N q]);
    Ystar = reshape(repmat(responses(1,:,:),[ttrial,1,1]),[ttrial*N q]);
    responses = permute(responses(1,:,:),[2 3 1]); % N x q
end

max_num_classes = 5;
classification = length(unique(responses(:))) < max_num_classes;

if classification
    Ycopy = Y;
    if size(Ycopy,1) == N 
        Ycopy = repmat(reshape(Ycopy,[1 N q]),[ttrial 1 1]);
    end
    % no demeaning by default if this is a classification problem
    if ~isfield(options,'demeanstim'), options.demeanstim = 0; end
end

if ~all(Y(:)==Ystar(:))
    error('For cross-validating, the same stimulus must be presented for the entire trial'); 
end

% Preproc data and put in the right format 
if do_preproc
    [X,Y,T,options] = preproc4hmm(X,Y,T,options); % this demeans Y
    p = size(X,2);
end

if isfield(options,'CVmethod') 
    CVmethod = options.CVmethod; options = rmfield(options,'CVmethod');
else
    CVmethod = 1;
end
if isfield(options,'NCV')
    NCV = options.NCV; options = rmfield(options,'NCV');
else, NCV = 10; 
end
if isfield(options,'lambda') 
    lambda = options.lambda; options = rmfield(options,'lambda');
else, lambda = 0.0001; 
end
if isfield(options,'Nfeatures') && options.Nfeatures>0 && options.Nfeatures<p
    warning('Specifying Nfeatures can lead to a biased calculation of CV-R2')
end
if isfield(options,'verbose') 
    verbose = options.verbose; options = rmfield(options,'verbose');
else, verbose = 1; 
end

if ~isfield(options,'c')
    if classification
        %disp('Response is treated as categorical')
        tmp = zeros(N,1);
        for j = 1:q
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
        c2 = cvpartition(N,'KFold',NCV);
        %disp('Response is treated as continuous - no CV stratification')
    end
    c = struct();
    c.test = cell(NCV,1);
    c.training = cell(NCV,1);
    for icv = 1:NCV
        c.training{icv} = find(c2.training(icv));
        c.test{icv} = find(c2.test(icv));
    end; clear c2
else
    c = options.c; options = rmfield(options,'c');
end

X = reshape(X,[ttrial N p]);
Y = reshape(Y,[ttrial N q]);
RidgePen = lambda * eye(p);

% Get Gamma and the Betas for each fold
Gammapred = zeros(ttrial,N,K); Betas = zeros(p,q,K,NCV); 
for icv = 1:NCV
    Ntr = length(c.training{icv}); Nte = length(c.test{icv});
    Xtrain = reshape(X(:,c.training{icv},:),[Ntr*ttrial p] ) ;
    Ytrain = reshape(Y(:,c.training{icv},:),[Ntr*ttrial q] ) ;
    Ttr = T(c.training{icv});
    [tuda,Gammatrain] = call_tudatrain(Xtrain,Ytrain,Ttr,options);
    Betas(:,:,:,icv) = tudabeta(tuda);
    switch CVmethod
        case 1 % training average
            mGammatrain = squeeze(mean(reshape(Gammatrain,[ttrial Ntr K]),2)); 
            for j = 1:Nte, Gammapred(:,c.test{icv}(j),:) = mGammatrain; end
        case 2 % regression
            Xtest = permute(X(:,c.test{icv},:),[2 3 1]);
            for t = 1:ttrial
                B = (Xtrain(:,:,t)' * Xtrain(:,:,t) + RidgePen) \ ...
                    Xtrain(:,:,t)' * Gammatrain(:,:,t);
                pred = Xtest(:,:,t) * B;
                pred = pred - repmat(min(min(pred,[],2), zeros(Nte,1)),1,K);
                pred = pred ./ repmat(sum(pred,2),1,K);
                Gammapred(t,c.test{icv},:) = pred;
            end
    end          
end

% Perform the prediction 
Ypred = zeros(ttrial,N,q);
for icv = 1:NCV
    Nte = length(c.test{icv});
    Xtest = reshape(X(:,c.test{icv},:),[ttrial*Nte p]);
    Gammatest = reshape(Gammapred(:,c.test{icv},:),[ttrial*Nte K]);
    for k = 1:K
        sGamma = repmat(Gammatest(:,k),[1 q]);
        Ypred(:,c.test{icv},:) = Ypred(:,c.test{icv},:) + ...
            reshape( (Xtest * Betas(:,:,k,icv)) .* sGamma , [ttrial Nte q]);
    end
end

if classification
    Y = reshape(Y,[ttrial*N q]);
    Y = round(Y); % get rid of noise we might have injected 
    Ypred = reshape(Ypred,[ttrial*N q]);
    Ypred_star = reshape(continuous_prediction_2class(Ycopy,Ypred),[ttrial N q]);
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
        tmp = sum(abs(Y - Ypred_star),2) < 1e-4;
    end
    acc = mean(tmp);
    acc_star = squeeze(mean(reshape(tmp, [ttrial N q]),2));
else   
    Ypred_star = Ypred; 
    Ypred = permute( mean(Ypred_star,1) ,[2 3 1]);
    % acc is explained variance 
    acc = sum( (response - Ypred).^2 ) ./ sum(response.^2) ; 
    acc_star = zeros(ttrial,q); 
    for t = 1:ttrial
        y = permute(Y(t,:,:),[2 3 1]); 
        acc_star(t,:) = 1 - sum((y - permute(Ypred_star(t,:,:),[2 3 1])).^2) ./ sum(y.^2);
    end
    Ypred_star = reshape(Ypred_star, [ttrial*N q]);
end
    
end


function [tuda,Gamma] = call_tudatrain(X,Y,T,options)

N = length(T); q = size(Y,2); p = size(X,2);

% Put X and Y together
Ttmp = T;
T = T + 1;
Z = zeros(sum(T),q+p,'single');
for n=1:N
    t1 = (1:T(n)) + sum(T(1:n-1));
    t2 = (1:Ttmp(n)) + sum(Ttmp(1:n-1));
    Z(t1(1:end-1),1:p) = X(t2,:);
    Z(t1(2:end),(p+1):end) = Y(t2,:);
end 

options = rmfield(options,'parallel_trials');
if isfield(options,'add_noise'), options = rmfield(options,'add_noise'); end

% Run TUDA inference
options.S = -ones(p+q);
options.S(1:p,p+1:end) = 1;
% 1. Estimate state time courses, trans prob mat, and first approximation
%       of the decoding models
options.updateObs = 0;
options.updateGamma = 1;
[~,Gamma] = hmmmar(Z,T,options);
% 2. Update state distributions only, leaving fixed the state time courses
options.updateObs = 1;
options.updateGamma = 0;
options.Gamma = Gamma;
options.tuda = 1;
%options.hmm = tuda; 
options.tudamonitoring = 0;
options.behaviour = [];
options.verbose = 0;
tuda = hmmmar(Z,T,options); 

end
