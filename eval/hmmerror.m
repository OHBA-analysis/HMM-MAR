function [errY,fracerrY,errR,fracerrR,response]=hmmerror(X,T,hmm,Gamma,test,residuals,actstates)
%
% Computes the error and explained variance for the given data points
%
% INPUT
% X         Observations
% T         Number of time points for each time series
% hmm       hmm data structure
% Gamma     probability of current state cond. on data -
%           inference is run for time points with Gamma=NaN,
% test      sum(T) x 1 vector indicating in which time points the error is to be computed;
%           common use will be to set test(t) = 1 when Gamma is NaN at this point
% residuals     in case we train on residuals, the value of those.
% actstates     Kx1 vector indicating which states were effectively used in the training
%
% OUTPUT
% errY             mean quadratic error
% fracerrY       fractional quadratic error
% errR             mean quadratic error of the residuals
% fracerrR       fractional quadratic error of the residuals
% response      mean of the predictive response distribution for test(t)=1
%
% Author: Diego Vidaurre, OHBA, University of Oxford

N = length(T);
train = hmm.train;
orders = formorders(train.order,train.orderoffset,train.timelag,train.exptimelag);
S = hmm.train.S==1;
Sind = formindexes(orders,train.S);
regressed = sum(S,1)>0;

if nargin<5,
    test = ones(sum(T),1);
end
if nargin<6 || isempty(residuals),
    residuals = ...
        getresiduals(X,T,Sind,train.maxorder,train.order,train.orderoffset,train.timelag,train.exptimelag,train.zeromean);
end
if nargin<7,
    actstates = ones(hmm.K,1);
end

Y = [];
te = [];
for in=1:N
    t0 = sum(T(1:in-1));
    Y = [Y; X(t0+1+hmm.train.maxorder:t0+T(in),:)];
    te = [te; test(t0+1+hmm.train.maxorder:t0+T(in))];
end

[response,responseR] = hmmpred(X,T,hmm,Gamma,residuals,actstates); 
response = response(te==1,:); responseR = responseR(te==1,:);
errY = mean((Y(te==1,regressed) - response(:,regressed)).^2);
fracerrY = errY ./ mean( (Y(te==1,regressed) - repmat(mean(Y(te==1,regressed)),sum(te),1) ).^2 );
errR = mean((residuals(te==1,regressed) - responseR(:,regressed)).^2);
fracerrR = errR ./ mean( (residuals(te==1,regressed) - repmat(mean(residuals(te==1,regressed)),sum(te),1) ).^2 );



