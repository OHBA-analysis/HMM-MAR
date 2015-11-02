function [p,LL] = emclrgr(X,Y,C,ind,K,nu,tol,maxit,trace)
% An EM algorithm for clusterwise regression in time series.
%
% The initial cluster values are sampled according to the following stochastic process:
% 0. t = 1
% 1. A cluster k is selected by sampling from a multinomial distribution
% 2. A permanence time r is sampled from a poisson distribution of rate nu
% 3. from t to t+r-1 we set the cluster membership to k
% 4. if t<=T goto 1
%
% Input parameters
% X: inputs, TxN matrix
% Y: outputs, TxM matrix
% C: a class vector, TX1. Only those with NaN will be clusterized 
% ind: NxM logical matrix specificying which features participate on each output variable 
% K: number of clusters
% nu: initialisation parameter; default T/100
% tol: an effective zero; default 0.01
% maxit: maximum number of iterations; default 100
% trace: if TRUE, shows progress
%
% Output values
% p: TxK matrix of soft cluster assignments
% LL: Log likelihood of the model
%
% Author: Diego Vidaurre, University of Oxford


[T, M] = size(Y);
N = size(X,2);
p = zeros(T,K);
K0 = size(C,2); if isnan(K0), K0 = 0; end
unknown = isnan(C(:,1));
Tnan = sum(unknown);
allind1 = all(ind(:));
if isempty(ind), predicting = 1:M;
else predicting = find(sum(ind)>0);
end

if nargin<5, nu = T/100; end
if nargin<6, tol = 0.00001; end
if nargin<7, maxit = 100; end
if nargin<8, trace = 0; end
 
% random init
t=1;
while t<=T
    r = min(poissrnd(nu),T-t+1);
    c = find(mnrnd(1,ones(1,K)/K ));
    p(t:min(t+r-1,T),c) = 1;
    t = t+r;
end
for k=1:K0 % overwrite what is fixed
    p(C==k,:) = 0; p(C==k,k) = 1; 
end

if N>0
    beta = zeros(N,M,K);
else
    beta = [];
end
omega = zeros(M,K);
for k=1:K
    if N>0
        if allind1
            beta(:,:,k) = pinv(X .* repmat(sqrt(p(:,k)),1,N)) * Y; 
        else
            for m=predicting
                beta(ind(:,m),m,k) = pinv(X(:,ind(:,m)) .* repmat(sqrt(p(:,k)),1,sum(ind(:,m)))) * Y(:,m); 
            end
        end
        %beta(:,:,k) = ( ((X' .* repmat(p(:,k)',N,1) ) * X) + 0.001 * eye(size(X,2)) ) \ ( (X' .* repmat(p(:,k)',N,1) ) * Y );
        er = sqrt(repmat(p(:,k),1,length(predicting) )) .* (Y(:,predicting) - X * beta(:,predicting,k));
    else
        er = sqrt(repmat(p(:,k),1,M)) .* Y;
    end
    omega(predicting,k) = sum(er.^2 ) / sum(p(:,k));
end
pi = sum(p) / sum(p(:));

LL0 = -Inf;
if isempty(beta)
    LL = getLL(X,Y(:,predicting),[],omega(predicting,:),pi);
else
    LL = getLL(X,Y(:,predicting),beta(:,predicting,:),omega(predicting,:),pi);
end

it = 0; p0 = p;
% EM   
while 1
    
    %if it>2 && LL-LL0 > tol*abs(LL), break; end
    if it>2 && LL0>LL, warning('error in the initialization - likelihood decreasing\n'); end
    
    if trace==1, fprintf('Iteration %d, LL: %f \n',it,LL); end
    LL0 = LL;
    
    % E step
    for k=1:K
        if N>0
            pred = X(unknown,:) * beta(:,predicting,k);
        else
            pred = zeros(Tnan,M);
        end
        p(unknown,k) = pi(k) * prod(normpdf( Y(unknown,predicting) , pred,  ...
                repmat( sqrt(omega(predicting,k)'), length(unknown), 1)  ), 2) ; 
    end
    p = p ./ repmat(sum(p,2),1,K);
    if any(isnan(p(:)))
        %warning(sprintf('There may be numerical precision issues - a proportion of %f are NaN\n',sum(isnan(p(:))) / length(p(:))));
        warning('There may be numerical precision issues')
        p(isnan(p(:))) = 1/K; 
    end
    pi = mean(p);
    
    % M step
    for k=1:K
        if N>0
            if allind1
                beta(:,:,k) = ( ((X' .* repmat(p(:,k)',N,1) ) * X) ) \ ( (X' .* repmat(p(:,k)',N,1) ) * Y );
            else
                for m=predicting
                    beta(ind(:,m),m,k) = ( ((X(:,ind(:,m))' .* repmat(p(:,k)',sum(ind(:,m)),1) ) * X(:,ind(:,m))) ) \ ...
                        ( (X(:,ind(:,m))' .* repmat(p(:,k)',sum(ind(:,m)),1) ) * Y(:,m) );
                end
            end
            er = sqrt(repmat(p(:,k),1,length(predicting))) .* (Y(:,predicting) - X * beta(:,predicting,k));
        else
            %mu = sum(repmat(p(:,k),1,M) .* Y) / sum(p(:,k));
            er = sqrt(repmat(p(:,k),1,M)) .* Y;
        end
        omega(predicting,k) = sum( er.^2 ) / sum(p(:,k));
    end
    
    % LL calculation
    if isempty(beta)
        LL = getLL(X,Y(:,predicting),[],omega(predicting,:),pi);
    else
        LL = getLL(X,Y(:,predicting),beta(:,predicting,:),omega(predicting,:),pi);
    end
    
    it = it + 1;
    if it>maxit || LL-LL0<tol
        break
    end
    
end
end


function LL = getLL(X,Y,beta,omega,pi)
T = size(X,1); K = size(omega,2);
LL = zeros(T,K); 
for k = 1:K
    if ~isempty(beta)
        pred = X * beta(:,:,k);
    else
        pred = zeros(size(Y));
    end
    LL(:,k) = logmvnpdf(Y,pred,diag(omega(:,k)))';
end
LL = sum(logsumexp(LL + repmat(log(pi),T,1),2));
end


% function LL = getLL(X,Y,beta,omega,pi)
% [T N] = size(X); K = size(beta,3); q = size(Y,2);
% LL = zeros(T,1);
% for k = 1:K
%     pred = X * beta(:,:,k);
%     LLk = ones(T,1);
%     for d = 1:q
%         LLk = LLk .* normpdf(Y(:,d),pred(:,d),sqrt(omega(d,k)));
%     end
%     LLk = pi(k) * LLk;
%     LL = LL + LLk;
% end
% LL = sum(log(LL));
% end