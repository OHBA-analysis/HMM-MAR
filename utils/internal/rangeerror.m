function [r] = rangeerror(X,T,maxorder,orders,Y,lambda)
% estimates the range of the error

if isempty(orders)
    r = range(X);
    
else
    if nargin<6
        lambda=0.01;
    end
    ndim=size(X,2);
    
    XX = formautoregr(X,T,orders,maxorder);
    
    if lambda==0
        W = pinv(XX) * Y;
    else
        W = (XX' * XX + lambda*eye(length(orders)*ndim)) \ XX' * Y;
    end
    r = Y - XX * W;
    %r = 0.5*sum(r.^2 );
    r = range(r);
    
end
