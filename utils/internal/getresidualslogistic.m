function [residuals,W] = getresidualslogistic (X,T,Ydim)
%
% Compute residuals - for logistic models this amounts to just saving the
% data Y values (class labels) being predicted.
%
% INPUT
% X             time series
% T             length of series
%
%
% OUTPUT
% residuals
%
% Author: Diego Vidaurre, OHBA, University of Oxford


N = length(T);
dimX = size(X,2)-Ydim;
order=1; %always the case for logistic models
for in=1:N
    t0 = sum(T(1:in-1));
    t = sum(T(1:in-1)) - (in-1)*order;
    residuals(t+1:t+T(in)-order,:) = X(t0+order+1:t0+T(in),dimX+1:dimX+Ydim);
end

% correct to be on +1,-1 range if not already:
if any(~(residuals(:)==1 | residuals(:)==-1 | residuals(:)==0))
    ME=MException('getresidualslogistic:Ybad','Y in wrong format; should be plus/minus one, or zero for multinomial options');
    throw(ME);
end

%     residuals = zeros(sum(T)-length(T)*maxorder,size(X,2)); 
%     for in=1:N
%         t0 = sum(T(1:in-1));
%         t = sum(T(1:in-1)) - (in-1)*maxorder;
%         residuals(t+1:t+T(in)-maxorder,:) = X(t0+maxorder+1:t0+T(in),:);
%     end
end
