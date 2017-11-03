function CovMats = getCovMats(X,T,maxlag,partial,Flips)
% Get the autocorrelation matrices up to lag maxlag, for each trial
N = length(T); ndim = size(X,2);
if nargin<5, Flips = zeros(N,ndim); end
CovMats = zeros(ndim,ndim,maxlag+1,N);
eps = 1e-8;
t0 = 0;
for in=1:N 
    t1 = t0 + T(in);
    Xin = X(t0+1:t1,:); t0 = t1;
    do_xcorr = (numel(Xin) < 1000000) && (ndim < 20);
    for i=1:ndim, if Flips(in,i)==1, Xin(:,i) = -Xin(:,i); end; end
    if do_xcorr
        r = xcorr(Xin,maxlag,'coeff'); % xcorr is extremely memory consuming
        for j = 0:maxlag
            CovMats(:,:,j+1,in) = reshape(r(maxlag-j+1,:),[ndim ndim]) ;
            if partial
                CovMats(:,:,j+1,in) = inv(CovMats(:,:,j+1,in) + eps*eye(ndim));
            end
            CovMats(:,:,j+1,in) = CovMats(:,:,j+1,in) - diag(diag(CovMats(:,:,j+1,in)));
        end
    else
        r = lowmem_xcorr(Xin,maxlag);
        for j = 0:maxlag
            CovMats(:,:,j+1,in) = r(maxlag-j+1,:,:);
            if partial
                CovMats(:,:,j+1,in) = inv(CovMats(:,:,j+1,in) + eps*eye(ndim));
            end            
            CovMats(:,:,j+1,in) = CovMats(:,:,j+1,in) - diag(diag(CovMats(:,:,j+1,in)));
        end
    end
end
end

