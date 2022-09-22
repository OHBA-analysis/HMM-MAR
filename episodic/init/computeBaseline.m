function baseline = computeBaseline(options,X,T)

Sind = options.Sind == 1; 
lambda = options.ehmm_regularisation_baseline;
if nargin == 1
    XX = formautoregr(options.ehmm_baseline_data.X,options.ehmm_baseline_data.T,...
        options.orders,options.maxorder,options.zeromean);
    Y = getresiduals(options.ehmm_baseline_data.X,options.ehmm_baseline_data.T,...
        options.S,options.maxorder,options.order,...
        options.orderoffset,options.timelag,options.exptimelag,options.zeromean);
else
    XX = formautoregr(X,T,...
        options.orders,options.maxorder,options.zeromean);
    Y = getresiduals(X,T,...
        options.S,options.maxorder,options.order,...
        options.orderoffset,options.timelag,options.exptimelag,options.zeromean);
end
ndim = size(Y,2); np = size(XX,2);
gram = (XX(:,Sind)' * XX(:,Sind));
gram = (gram + gram') / 2 ;
gram = gram + lambda * eye(size(gram,2));
igram = inv(gram);
Mu_W = igram * (XX' * Y);
iS_W = zeros(ndim,np,np);
S_W = zeros(ndim,np,np);
for n = 1:ndim
    iS_W(n,Sind,Sind) = gram;
    S_W(n,Sind,Sind) = igram;
end
baseline = struct();
baseline.Mu_W = Mu_W;
baseline.iS_W = iS_W;
baseline.S_W = S_W;

end
