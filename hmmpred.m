function [responseY,responseR,Gamma] = hmmpred(X,T,hmm,Gamma,residuals,actstates)
%
% Predictive distribution of the response and error on potentially unseen data
% useful for cross-validation routines
%
% X         observations
% T         Number of time points for each time series
% hmm       hmm data structure
% Gamma     probability of current state cond. on data - 
%           inference is run for time points with Gamma=NaN,  
% residuals     in case we train on residuals, the value of those.
% actstates     Kx1 vector indicating which states were effectively used in the training, 
%               Gamma is assumed to have as many columns as initial states
%               were specified, so that sum(actstates)<=size(Gamma,2).
%               The default is ones(K,1)
%
% responseY mean of the predictive response 
% responseR mean of the predictive response for the residuals
% Gamma     estimated probability of current state cond. on data
%
% Author: Diego Vidaurre, OHBA, University of Oxford

K = hmm.K; ndim = size(X,2);
train = hmm.train;
[orders,order] = formorders(train.order,train.orderoffset,train.timelag,train.exptimelag);

if nargin<5 || isempty(residuals),
    [residuals,Wgl] = getresiduals(X,T,train.Sind,train.order.maxorder,train.order,...
        train.orderoffset,train.timelag,train.exptimelag,train.zeromean);
else
    Wgl = zeros(length(orders)*ndim+(~hmm.train.zeromean),ndim);
end
if nargin<6,
    actstates = ones(hmm.K,1);
end

if K<length(actstates), % populate hmm with empty states up to K
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

if any(isnan(Gamma)),
    data.X = X; data.C = [];
    for in=1:length(T);
        if in==1, s0 = 0; else s0 = sum(T(1:in-1)) - order*(in-1); end
        data.C = [data.C; NaN(order,K); Gamma(s0+1:s0+T(in)-order,:)];
    end
    Gamma=hsinference(data,T,hmm,residuals);
end

setxx; % build XX 

responseR = zeros(size(XX{1},1), ndim);
responseY = zeros(size(XX{1},1), ndim);
for k=1:K,
    if hmm.train.multipleConf, 
        kk = k;
        if isfield(hmm.state(k),'train') && hmm.state(k).train.uniqueAR
            W = repmat(hmm.state(k).W.Mu_W,1,ndim);
        else
            W = hmm.state(k).W.Mu_W;
        end
    else
        kk = 1; W = hmm.state(k).W.Mu_W;
    end
    if actstates(k)
        responseR = responseR + repmat(Gamma(:,acstates1==k),1,ndim) .*  (XX{kk} * W);
    end
    if isempty(Wgl)
        responseY = responseR;
    else
        responseY = responseR + XX{kk} * Wgl;
    end
end;
