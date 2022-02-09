function Xi = approximateXi_ehmm(Gamma,T,par)
if nargin==3
    if ~isstruct(par)
        order = par;
    else
        [~,order] = formorders(par.train.order,par.train.orderoffset,...
            par.train.timelag,par.train.exptimelag);
    end
else
    order = 0;
end
K = size(Gamma,2);
Xi = zeros(sum(T-1-order),K,2,2);
for j = 1:length(T)
    indG = (1:(T(j)-order)) + sum(T(1:j-1)) - (j-1)*order;
    indXi =  (1:(T(j)-order-1)) + sum(T(1:j-1)) - (j-1)*(order+1);
    for k = 1:K
        g = [Gamma(indG,k) (1-Gamma(indG,k))];
        for t = 1:length(indXi)
            xi = g(t,:)' * g(t+1,:);
            xi = xi / sum(xi(:));
            Xi(indXi(t),k,:,:) = xi;
        end
    end
end

end