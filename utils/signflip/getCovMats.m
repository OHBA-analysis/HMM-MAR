function CovMats = getCovMats(X,T,maxlag,partial,Flips)
% Get the autocorrelation matrices up to lag maxlag, for each trial
N = length(T); ndim = size(X,2);
if nargin<5, Flips = zeros(N,ndim); end
CovMats = zeros(ndim,ndim,2*maxlag+1,N);
eps = 1e-8;
t0 = 0;
for j = 1:N 
    t1 = t0 + T(j);
    Xj = X(t0+1:t1,:); t0 = t1;
    for i = 1:ndim, if Flips(j,i)==1, Xj(:,i) = -Xj(:,i); end; end
    CovMats(:,:,:,j) = lowmem_xcorr(Xj,maxlag);
    for l = 1:2*maxlag+1
        if partial
            CovMats(:,:,l,j) = inv(CovMats(:,:,l,j) + eps*eye(ndim));
        end
        CovMats(:,:,l,j) = CovMats(:,:,l,j) - diag(diag(CovMats(:,:,l,j)));
        % don't care about within channel autocorrelations
    end
end
end

