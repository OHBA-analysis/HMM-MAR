function [W,covm,pred,residuals,fracerr] = mlmar (X,T,Sind,maxorder,order,orderoffset,timelag,exptimelag,zeromean,W,Gamma)
%
% Estimates maximum-likelihood (ML) MAR model 
%
% INPUT
% X             observations
% T             length of series
% order,orderoffset,timelag,exptimelag,zeromean - check documentation
% W             MAR parameters - not to be computed if they are supplied
% Gamma         weights
%
% OUTPUT
% W             ML MAR parameters, (orderxN) by N
% covm          ML covariance matrix of the error
% pred          predicted values
% residuals     fit residuals
% fracerr       fractional error
%
% Author: Diego Vidaurre, OHBA, University of Oxford

ndim = size(X,2);
if nargin<6, orderoffset=0; end
if nargin<7, timelag=1; end
if nargin<8, exptimelag=1; end
if nargin<9, zeromean=1; end
if nargin<10, W = []; end
if nargin<11, Gamma = []; end

[orders,order] = formorders(order,orderoffset,timelag,exptimelag);
[XX,Y] = formautoregr(X,T,orders,maxorder,zeromean);

if isempty(Sind)
    Sind = true(ndim*length(orders),ndim);
end
if ~zeromean
    Sind = [true(1,size(X,2)); Sind];
end

if ~isempty(Gamma)
    if order>0, XX = repmat(sqrt(Gamma),1,size(XX,2)) .* XX; end
    Y = repmat(sqrt(Gamma),1,size(Y,2)) .* Y;
end

if isempty(W)
    if order>0 
        if all(Sind(:)) 
            W = XX \ Y; 
        else
            W = zeros(ndim*length(orders),ndim);
            for n=1:ndim
                %ind = n + (0:length(orders)-1) * ndim;
                if any(Sind(:,n))
                    W(Sind(:,n),n) = XX(:,Sind(:,n)) \ Y(:,n);
                end
            end
        end
        pred = XX * W;
    else
        pred = zeros(sum(T),ndim);
        W = [];
    end
else
    pred = XX * W;
end

residuals = Y - pred;
SSE = sum ( residuals.^2 );
covm = (residuals' * residuals ) / (size(Y,1)-1) ;
fracerr = SSE ./ sum( (Y - repmat( mean(Y),size(Y,1),1) ).^2 );


