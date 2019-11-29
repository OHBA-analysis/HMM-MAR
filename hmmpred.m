function [responseY,responseR,Gamma,explained_var] = ...
    hmmpred(X,T,hmm,Gamma,residuals,actstates,grouping)
%
% Predictive distribution of the response and error on potentially unseen data
% useful for cross-validation routines; that is, this function returns the
% mean predicted signal according to the HMM observation parameters
%
% X         observations
% T         Number of time points for each time series
% hmm       hmm data structure
% Gamma     probability of current state cond. on data - 
%           inference is run for time points with Gamma=NaN.
%           (It needs to have the same size of the state time courses as
%           would be inferred by hmmmar - no. of time points by states).
% residuals     in case we train on residuals, the value of those.
% actstates     Kx1 vector indicating which states were effectively used in the training, 
%               Gamma is assumed to have as many columns as initial states
%               were specified, so that sum(actstates)<=size(Gamma,2).
%               The default is ones(K,1)
%
% responseY mean of the predictive response 
% responseR mean of the predictive response for the residuals. This is
%           equal to responseY unless a global (state-independent) model is 
%           specified (which is not a default option). 
% Gamma     estimated probability of current state cond. on data
%
% Author: Diego Vidaurre, OHBA, University of Oxford

% to fix potential compatibility issues with previous versions
hmm = versCompatibilityFix(hmm); 

if nargin<7 || isempty(grouping) 
    if isfield(hmm.train,'grouping')
        grouping = hmm.train.grouping;
    else
        grouping = ones(length(T),1);
    end
    if size(grouping,1)==1,  grouping = grouping'; end
end
hmm.train.grouping = grouping;

K = hmm.K; ndim = size(X,2);
train = hmm.train;
[orders,order] = formorders(train.order,train.orderoffset,train.timelag,train.exptimelag);

if nargin<5 || isempty(residuals)
    train.Sind = formindexes(orders,train.S);
    [residuals,Wgl] = getresiduals(X,T,train.Sind,train.maxorder,train.order,...
        train.orderoffset,train.timelag,train.exptimelag,train.zeromean);
else
    Wgl = zeros(length(orders)*ndim+(~hmm.train.zeromean),ndim);
end
if nargin<6
    actstates = ones(hmm.K,1);
end

if K<length(actstates) % populate hmm with empty states up to K
    hmm2 = hmm; hmm2 = rmfield(hmm2,'state');
    acstates1 = find(actstates==1);
    if strcmp(train.covtype,'diag') || strcmp(train.covtype,'full')    
        omegashape = 0;
        if strcmp(train.covtype,'diag'), omegarate = zeros(1,ndim);
        else omegarate = zeros(ndim); end
        for k=1:K
            omegashape = omegashape + hmm.state(k).Omega.Gam_shape / K;
            omegarate = omegarate + hmm.state(k).Omega.Gam_rate / K;
        end
        if strcmp(train.covtype,'diag'), iomegarate = omegarate.^(-1);
        else iomegarate = inv(omegarate); end
    end
    W = zeros((~train.zeromean)+ndim*length(orders),ndim);
    S_W = zeros(ndim,(~train.zeromean)+ndim*length(orders),(~train.zeromean)+ndim*length(orders));
    for k=1:length(actstates)
        if actstates(k)==1
            hmm2.state(k) = struct('Omega',hmm.state(acstates1==k).Omega,'W',hmm.state(acstates1==k).W);
        else
            hmm2.state(k) = struct('Omega',struct('Gam_shape',omegashape,'Gam_rate',omegarate,'Gam_irate',iomegarate),...
                'W',struct('Mu_W',W,'S_W',S_W));
        end
    end
    K = length(actstates);
    hmm = hmm2; clear hmm2; 
else
    acstates1 = 1:K;
end

if any(isnan(Gamma))
    data.X = X; data.C = [];
    for n = 1:length(T)
        if n==1, s0 = 0; else s0 = sum(T(1:n-1)) - order*(n-1); end
        data.C = [data.C; NaN(order,K); Gamma(s0+1:s0+T(n)-order,:)];
    end
    Gamma = hsinference(data,T,hmm,residuals);
end

setxx; % build XX 

responseR = zeros(size(XX,1), ndim);
responseY = zeros(size(XX,1), ndim);
for k = 1:K
    W = hmm.state(k).W.Mu_W;
    if actstates(k)
        responseR = responseR + repmat(Gamma(:,acstates1==k),1,ndim) .*  (XX * W);
    end
    if isempty(Wgl)
        responseY = responseR;
    else
        responseY = responseR + XX * Wgl;
    end
end

explained_var = 1 - sum((responseY - residuals).^2) ./ sum((residuals - mean(residuals)).^2);

end