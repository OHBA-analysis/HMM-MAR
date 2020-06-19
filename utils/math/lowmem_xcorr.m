function C = lowmem_xcorr(X,maxlag)

T = size(X,1); ndim = size(X,2);
Y = embeddata(X,T,-maxlag:maxlag);
CY = corr(Y);
L = 2*maxlag+1;
I = false(L); for j = 1:L, I(j,L+1-j) = true; end
C = zeros(ndim,ndim,L);

for n1 = 1:ndim
    ind1 = (1:L) + (n1-1)*L;
    for n2 = 1:ndim
        ind2 = (1:L) + (n2-1)*L;
        cy = CY(ind1,ind2); 
        C(n1,n2,:) = cy(I);
    end
end     
            
end
