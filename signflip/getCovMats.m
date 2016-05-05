function CovMats = getCovMats(X,T,maxlag,Flips)
% Get the autocorrelation matrices up to lag maxlag, for each trial
N = length(T); ndim = size(X,2);
if nargin<4, Flips = zeros(N,ndim); end
CovMats = zeros(ndim,ndim,maxlag+1,N);
t0 = 0;
for in=1:N 
    t1 = t0 + T(in);
    Xin = X(t0+1:t1,:); t0 = t1;
    for i=1:ndim, if Flips(in,i)==1, Xin(:,i) = -Xin(:,i); end; end
    r = xcorr(Xin,maxlag,'coeff'); % xcorr is extremely memory consuming    
    for j = 0:maxlag
        CovMats(:,:,j+1,in) = reshape(r(maxlag-j+1,:),[ndim ndim]) ;
        CovMats(:,:,j+1,in) = CovMats(:,:,j+1,in) - diag(diag(CovMats(:,:,j+1,in)));
    end    
end
end