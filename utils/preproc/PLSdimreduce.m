function [newaxes,A] = PLSdimreduce(X,Y,T,numdim)
%note this function just takes data matrix X and design matrix Y and
%outputs a new set of axes for the data X that maximises the covariance of
%X and Y. Note that newaxes = X * A

if ~all(T==T(1))
    error('PLS dim reduction can only be applied where all trials have same length');
end
q = size(Y,2);
B = [];
for t=1:T(1)
    [~,~,XS] = plsregress(X(t:T(1):end,:),Y(t:T(1):end,:),q);
    B = [B,pinv(X(t:T(1):end,:))*XS];
end
[~,B_pls] = pca(B,'Centered',false);
A = B_pls(:,1:numdim);
sig = std(X*A);
A = A*diag(sig.^-1);
newaxes = X*A;

end