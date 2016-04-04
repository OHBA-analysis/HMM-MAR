function [pval,sig,gc_time,gc_freq] = tgrangerc (X,T,Gamma,options)
%
% Computes transient/state-wise Granger causality
% needs mvgc (http://www.sussex.ac.uk/sackler/mvgc/) in the path 
% 
% INPUT
% X             Time series
% T             length of series
% Gamma         Posterior probability of the states given the data
% options
%  .maxorder    If positive, the best order (according to corrected AIC) is selected up to maxorder
%                   if negative, the order is fixed to -maxorder. If empty, it
%                   uses hmm.train.maxorder
%  .alpha       Significance value (default 0.05)
%  .Fs          Frequency resolution
%  .mvgc_path   path to the mvgc toolbox
%  .verbose     
%  .showplot        
%
% OUTPUT
% C             Causality array K X ndim X ndim, where element (k,i,j) is 1 if i causes j in state k, and 0 otherwise
% fstat         array of F-stat values - if F(k,i,j) > critval for state k, we reject the null hypothesis of no-casuality
% critval       critical value of the F-distribution
% err0          Matrix K x ndim or quadratic errors
% err1          Array K X ndim X ndim of sum of quadratic errors, without
%               the channel indicated in the second dimension
%
% Author: Diego Vidaurre (2015), OHBA, University of Oxford

order_hmm = (size(X,1)-size(Gamma,1))/length(T);

if ~isfield(options,'mvgc_path'), error('You need to provide the path to mvgc toolbox'); end
if ~isfield(options,'maxorder'), maxorder = order_hmm;
else maxorder = options.maxorder;
end
if ~isfield(options,'verbose'), verbose = 1;
else verbose = options.verbose;
end
if ~isfield(options,'showplot'), showplot = 0;
else showplot = options.showplot;
end
if ~isfield(options,'alpha'), alpha = 0.05;
else alpha = options.alpha;
end
if ~isfield(options,'Fs'), Fs = 1;
else Fs = options.Fs;
end

init_mvgc;

K=size(Gamma,2);
ndim = size(X,2);
N = length(T);

pval = cell(K,1); sig = cell(K,1); F = cell(K,1);


if maxorder > 0 % model selection
    if verbose, fprintf('Using AIC for model selection \n'); end
    aic = zeros(maxorder,1);
    for order=1:maxorder
        if verbose, fprintf('Trying out order %d ... ',order); end
        if order_hmm>order % trim X
            d = order_hmm-order;
            T2 = T - d;
            X2 = zeros(sum(T2),ndim);
            for in=1:N
                t0 = sum(T(1:in-1));
                t00 = sum(T(1:in-1)) - d*(in-1);
                X2(t00+1:t00+T(in)-d,:) = X(t0+1+d:t0+T(in),:);
            end
            Gamma2 = Gamma;
        elseif order_hmm<order % trim Gamma
            d = order - order_hmm;
            Gamma2 = zeros(sum(T)-length(T)*order,K);
            for in=1:N
                t0 = sum(T(1:in-1)) - order_hmm*(in-1);
                t00 = sum(T(1:in-1)) - order*(in-1);
                Gamma2(t00+1:t00+T(in)-order,:) = Gamma(t0+1+d:t0+T(in)-order_hmm,:);
            end
            X2 = X; T2 = T;
        else % equal
            X2 = X; T2 = T;
            Gamma2 = Gamma;
        end
        for k=1:K
            [~,~,~,res] = mlmar(X2,T2,[],order,order,0,1,[],0,[],Gamma2(:,k)); % MAR model estimation
            %df = sum(Gamma2(:,k));
            %for n=1:ndim
            %    aic(order,k) = aic(order,k) + df * log(sum( residuals(:,n).^2 ) ) + 2 * ndim * order;
            %end
            m = sum(Gamma2(:,k));  
            w = order * ndim;
            aic(order,k) = m*log(det((res'*res)/(m-1))) + 2*w*(m/(m-w-1));
            if k==1
                fprintf('%d: %g = %g + %g \n',order,aic(order,k), m*log(det((res'*res)/(m-1))), 2*w*(m/(m-w-1)))
            end
            if verbose, fprintf(' AIC = %g \n',aic(order,k)); end
        end
    end
    [~,order] = min(sum(aic,2));
    if verbose, fprintf('We choose order %d \n',order); end
else
    order = -maxorder;
end

for k=1:K
    [W,covm] = mlmar(X2,T2,[],order,order,0,1,[],0,[],Gamma2(:,k)); % MAR model estimation
    W = permute(reshape(W(2:end,:),ndim,order,ndim),[1 3 2]);
    G = var_to_autocov(W,covm); % Autocovariance calculation
    gc_time{k} = autocov_to_pwcgc(G); % Granger causality calculation: time domain
    gc_freq{k} = autocov_to_spwcgc(G,Fs); % Granger causality calculation: freq domain
    pval{k} = state_mvgc_pval(gc_time{k},order,sum(Gamma2(:,k)),ndim,ndim);
    sig{k}  = significance(pval{k},alpha,'FDR');
    if showplot, figure(k); plot_spw(gc_freq{k},Fs); end
end

end


function pval = state_mvgc_pval(x,p,m,nx,ny)

pval = NaN(size(x)); % output p-value matrix is same shape as x matrix
nn   = ~isnan(x);    % indices of non-NaN x values (logical array)
x    = x(nn);        % vectorise non-NaN x values
pval(nn) = 1-state_mvgc_cdf(x,p,m,nx,ny); % assume null hypothesis F = 0

end


function P = state_mvgc_cdf(x,p,m,nx,ny)

assert(isvector(x),'evaluation values must be a vector');
n = length(x);
P = zeros(n,1);

% Geweke chi2 test form
d = p*nx*ny;             
for i = 1:n
    P(i) = chi2cdf(m*x(i),d);
end

end





