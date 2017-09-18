function C = lowmem_xcorr(X,maxlag)
ndim = size(X,2);
C = zeros(2*maxlag+1,ndim,ndim);
for i=0:maxlag
    for n1=1:ndim
        for n2=1:ndim
            C(maxlag+1+i,n1,n2) = corr(X(1:end-i,n1),X(1+i:end,n2));
            C(maxlag+1-i,n1,n2) = C(maxlag+1+i,n1,n2);
        end
    end
end
end
