function sp = specificity(X,T,hmm,Gamma)
%
% Get a measure of how specific or distinct from each other are the states 
% 
% INPUT
% X             a matrix containing the time series
% T             length of series
% hmm           the HMM-MAR structure 
% Gamma         Time courses of the states probabilities given data
%
% OUTPUT
% sp            Value of the specificity measure
%
% Author: Diego Vidaurre, OHBA, University of Oxford

regressed = sum(hmm.train.S==1,1)>0;
var0 = sum( (X(:,regressed) - repmat(mean(X(:,regressed)),size(X,1),1)).^2 );
orders = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
Sind = formindexes(orders,hmm.train.S);

% get global residuals
[~,~,~,r0] = mlmar(X,T,Sind==1,hmm.train.maxorder,hmm.train.order,hmm.train.orderoffset, ...
    hmm.train.timelag,hmm.train.exptimelag,hmm.train.zeromean);
e0 = sum(sum( r0(:,regressed).^2 ) ./ var0);

% get local residuals
[~,~,gcovm] = mlhmmmar(X,T,hmm,Gamma);
nsamples = sum(T) - length(T)*hmm.train.maxorder;
e2 = sum( (diag(gcovm)' * nsamples) ./ var0);
sp = e2/e0;





