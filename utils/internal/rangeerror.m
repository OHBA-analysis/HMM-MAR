function r = rangeerror(X,T,Y,orders,options)
% estimates the range of the error

maxorder = options.maxorder; 
S = options.S;
Sind = options.Sind == 1;
lambda = 0.01;

if isempty(orders)
    r = range(X - repmat(mean(X),size(X,1),1));
else
    ndim = size(X,2);
    XX = formautoregr(X,T,orders,maxorder,options.zeromean);
    r = zeros(1,ndim);
    W = zeros(size(XX,2),size(Y,2));
    for n = 1:ndim
        ndim_n = sum(S(:,n)>0);
        if ndim_n==0 && options.zeromean==1, continue; end
        W(Sind(:,n),n) = (XX(:,Sind(:,n))' * XX(:,Sind(:,n)) + lambda*eye(sum(Sind(:,n)))) ...
            \ XX(:,Sind(:,n))' * Y(:,n);
        r(n) = range(Y(:,n) - XX(:,Sind(:,n)) * W(Sind(:,n),n));
    end
end
