function [XX,Y] = formautoregr(X,T,orders,maxorder,zeromean,single_format,B,V)
%
% form regressor and response for the autoregression;
% residuals are assumed to have size T(in)-order in each trial 
% 
% Author: Diego Vidaurre, OHBA, University of Oxford
 
N = length(T); ndim = size(X,2);
if nargin<5, zeromean = 1; end
if nargin<6, single_format = 0; end
if nargin<7 || isempty(B), B = []; Q = ndim; 
else Q = size(B,2); end
if nargin<8 || isempty(V), V = []; M = 0;
else M = size(V,2); end

if M==0
    if single_format
        XX = zeros(sum(T)-N*maxorder,length(orders)*Q+(~zeromean),'single');
    else
        XX = zeros(sum(T)-N*maxorder,length(orders)*Q+(~zeromean));
    end
else
    if single_format
        XX = zeros(sum(T)-N*maxorder,M+(~zeromean),'single');
    else
        XX = zeros(sum(T)-N*maxorder,M+(~zeromean));
    end
end
if nargout==2
    Y = zeros(sum(T)-N*maxorder,ndim);
end

t_cumulative = cumsum([0;T(:)]);
for in=1:N
    t0 = t_cumulative(in); 
    s0 = t0 - maxorder*(in-1);
    if nargout==2
        Y(s0+1:s0+T(in)-maxorder,:) = X(t0+maxorder+1:t0+T(in),:);
    end
    if ~isempty(V)
        if single_format
            XX_i = zeros(T(in)-maxorder,length(orders)*Q,'single'); 
        else
            XX_i = zeros(T(in)-maxorder,length(orders)*Q); 
        end
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
        elseif ~isempty(V)
            if single_format
                XX_i(:,(1:Q)+(i-1)*Q) = single(X(t0+maxorder-o+1:t0+T(in)-o,:));
            else
                XX_i(:,(1:Q)+(i-1)*Q) = X(t0+maxorder-o+1:t0+T(in)-o,:);
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
    if ~isempty(V) 
        if single_format
            XX(s0+1:s0+T(in)-maxorder,(1:M)+(~zeromean)) = XX_i * single(V);
        else
            XX(s0+1:s0+T(in)-maxorder,(1:M)+(~zeromean)) = XX_i * V;
        end
    end
end
if ~zeromean, XX(:,1) = 1; end
end