function expl_var = explainedvar_singleMAR(X,T,options)
% Tells how much variance (between 0 and 1) explains a (maximum-likelihood)
% MAR model on data X, parametrised by options
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2016)

ndim = size(X,2); N = length(T);

if ~isfield(options,'order'), error('order was not specified'); end
if ~isfield(options,'timelag'), options.timelag = 1; end
if ~isfield(options,'exptimelag'), options.exptimelag = 1; end
if ~isfield(options,'orderoffset'), options.orderoffset = 0; end

X = X - repmat(mean(X),sum(T),1);
X = X ./ repmat(std(X),sum(T),1);

orders = formorders(options.order,options.orderoffset,options.timelag,options.exptimelag);
XX = formautoregr(X,T,orders,options.order,1);
Y = zeros(sum(T)-length(T)*options.order,ndim);  
for n=1:N
    t0 = sum(T(1:n-1));
    t00 = sum(T(1:n-1)) - (n-1) * options.order;
    Y(t00+1:t00+T(n)-options.order,:) = X(t0+options.order+1:t0+T(n),:);
end

B = pinv(XX) * Y;
R = XX * B - Y;
expl_var = 1 - sum(R.^2) ./ sum(Y.^2);

end