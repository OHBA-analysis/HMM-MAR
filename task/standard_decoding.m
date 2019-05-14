function [cv_acc,acc,cv_err,err] = standard_decoding(X,Y,T,options,binsize)
% Compute cross-validated (CV) accuracies from a "decoding" model trained 
% at each time point (or window, see below) in the trial  
% The reported statistic is the cross-validated explained variance from
% regressing X on the stimulus.
%
% INPUT
% X: Brain data, (time by regions) or (time by trials by regions)
% Y: Stimulus, (time by q); q is no. of stimulus features OR
%              (no.trials by q), meaning that each trial has a single
%              stimulus value
% T: Length of series
% options: structure with the preprocessing options - see documentation in 
%                       https://github.com/OHBA-analysis/HMM-MAR/wiki
% binsize: how many consecutive time points will be used for each estiamtion. 
%           By default, this is 1, such that one decoding model
%               is estimated using 1 time point
%
% OUTPUT
% cv_acc:  (time points by stimuli) time series of CV-accuracies
% acc:     (time points by stimuli) time series of non-CV accuracies 
% cv_err:  (time points by stimuli) time series of CV-squared error
% err:     (time points by stimuli) time series of non-CV squared error 
% 
% Author: Diego Vidaurre, OHBA, University of Oxford (2018)

options.Nfeatures = 0; 
options.K = 1; 
[X,Y,T] = preproc4hmm(X,Y,T,options); % this demeans Y

N = length(T); ttrial = T(1); 
p = size(X,2); q = size(Y,2);
X = reshape(X,[ttrial N p]);
Y = reshape(Y,[ttrial N q]);

if (binsize/2)==0
    warning(['binsize must be an odd number, setting to ' num2str(binsize+1)])
    binsize = binsize + 1; 
end

if any(T~=ttrial), error('All trials must have the same length'); end
if nargin<5, binsize = 1; end

if isfield(options,'NCV')
    NCV = options.NCV; options = rmfield(options,'NCV');
else, NCV = 10; 
end
if isfield(options,'lambda') 
    lambda = options.lambda; options = rmfield(options,'lambda');
else, lambda = 0.0001; 
end
if isfield(options,'lossfunc') 
    lossfunc = options.lossfunc; options = rmfield(options,'lossfunc');
else, lossfunc = 'quadratic'; 
end

% Form CV folds; if response are categorical, then it's stratified
%samples_per_value = length(Y) / length(unique(Y));
%stratified = samples_per_value > 200;
stratified = length(unique(Y(:))) < 5;
responses = permute(Y(1,:,:),[2 3 1]);
if ~isfield(options,'c')
    if stratified
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
        c.training{icv} = c2.training(icv);
        c.test{icv} = c2.test(icv);
    end; clear c2
else
    c = options.c;  
end

halfbin = floor(binsize/2); 
%nwin = round(ttrial / binsize);
%binsize = floor(ttrial / nwin); 
RidgePen = lambda * eye(p);

Ypred = NaN(size(Y));
for icv = 1:NCV
    Ntr = sum(c.training{icv}); Nte = sum(c.test{icv});
    for t = halfbin+1 : ttrial-halfbin
        r = t-halfbin:t+halfbin;
        %r = (1:binsize) + (t-1)*binsize;
        Xtr = reshape(X(r,c.training{icv},:),binsize*Ntr,p);
        Ytr = reshape(Y(r,c.training{icv},:),binsize*Ntr,q);
        Xte = reshape(X(t,c.test{icv},:),Nte,p);
        beta = (Xtr' * Xtr + RidgePen) \ (Xtr' * Ytr);
        Ypred(t,c.test{icv},:) = reshape(Xte * beta,Nte,q);
    end
end

cv_acc = NaN(ttrial,1);
cv_err = NaN(ttrial,1); 
for t = halfbin+1 : ttrial-halfbin
    Yt = reshape(Y(t,:,:),N,q);
    Ypredt = reshape(Ypred(t,:,:),N,q);
    [cv_acc(t),cv_err(t)] = get_R2(Yt,Ypredt,lossfunc,1);
end

% non-cross validated
acc = NaN(ttrial,1);
err = NaN(ttrial,1); 
Ypred = zeros(size(Y));

for t = halfbin+1 : ttrial-halfbin
    r = t-halfbin:t+halfbin;
    Xt = reshape(X(r,:,:),binsize*N,p);
    Yt = reshape(Y(r,:,:),binsize*N,q);
    beta = (Xt' * Xt + RidgePen) \ (Xt' * Yt);
    Xt = reshape(X(t,:,:),N,p);
    Yt = reshape(Y(t,:,:),N,q);
    Ypred(t,:,:) = reshape(Xt * beta,N,q);
    Ypredt = permute(Ypred(t,:,:),[2 3 1]);
    [acc(t),err(t)] = get_R2(Yt,Ypredt,lossfunc,1);
end

end