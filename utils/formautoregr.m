function [XX,Y] = formautoregr(X,T,orders,maxorder,zeromean,single_format,B)
%
% form regressor and response for the autoregression;
% residuals are assumed to have size T(in)-order in each trial 
% 
% Author: Diego Vidaurre, OHBA, University of Oxford
 
N = length(T); ndim = size(X,2);
if nargin<5, zeromean = 1; end
if nargin<6, single_format = 0; end
if nargin<7, B = []; Q = ndim; 
else Q = size(B,2); end

if single_format
    XX = zeros(sum(T)-length(T)*maxorder,length(orders)*Q+(~zeromean),'single');
else
    XX = zeros(sum(T)-length(T)*maxorder,length(orders)*Q+(~zeromean));
end
if nargout==2
    Y = zeros(sum(T)-length(T)*maxorder,ndim);
end
for in=1:N
    t0 = sum(T(1:in-1)); s0 = sum(T(1:in-1)) - maxorder*(in-1);
    if nargout==2
        Y(s0+1:s0+T(in)-maxorder,:) = X(t0+maxorder+1:t0+T(in),:);
    end
    for i=1:length(orders)
        o = orders(i);
        if ~isempty(B)
            if single_format
                XX(s0+1:s0+T(in)-maxorder,(1:Q)+(i-1)*Q+(~zeromean)) = ...
                    single(X(t0+maxorder-o+1:t0+T(in)-o,:)) * single(B(:,:,i));
            else
                XX(s0+1:s0+T(in)-maxorder,(1:Q)+(i-1)*Q+(~zeromean)) = ...
                    X(t0+maxorder-o+1:t0+T(in)-o,:) * B(:,:,i); 
            end
        else
            if single_format
                XX(s0+1:s0+T(in)-maxorder,(1:ndim)+(i-1)*ndim+(~zeromean)) = ...
                    single(X(t0+maxorder-o+1:t0+T(in)-o,:));
            else
                XX(s0+1:s0+T(in)-maxorder,(1:ndim)+(i-1)*ndim+(~zeromean)) = ...
                    X(t0+maxorder-o+1:t0+T(in)-o,:);
            end
        end
    end
end
if ~zeromean, XX(:,1) = 1; end
end