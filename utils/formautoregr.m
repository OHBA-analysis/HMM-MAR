function [XX,Y] = formautoregr(X,T,orders,maxorder,zeromean)
%
% form regressor and response for the autoregression;
% residuals are assumed to have size T(in)-order in each trial 
% 
% Author: Diego Vidaurre, OHBA, University of Oxford
 
N = length(T); ndim = size(X,2);
if nargin<5, zeromean = 1; end
    
XX = []; Y = [];
for in=1:N
    t0 = sum(T(1:in-1)); 
    Y = [Y; X(t0+maxorder+1:t0+T(in),:)];
    XX0 = zeros(T(in)-maxorder,length(orders)*ndim);
    for i=1:length(orders)
        o = orders(i);
        XX0(:,(1:ndim) + (i-1)*ndim) = X(t0+maxorder-o+1:t0+T(in)-o,:);
    end;
    XX = [XX; XX0];
end
if ~zeromean, 
    XX = [ones(size(XX,1),1) XX];
end
