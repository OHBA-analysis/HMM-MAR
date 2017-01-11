function X = simgauss(T,hmm,Gamma)

if iscell(T)
    if size(T,1)==1, T = T'; end
    T = cell2mat(T);
end

ndim = size(hmm.state(1).W.Mu_W,2); K = size(Gamma,2);
X = zeros(T,ndim);
mu = zeros(T,ndim);

switch hmm.train.covtype
    case 'uniquediag'
        Std = sqrt(hmm.Omega.Gam_rate / hmm.Omega.Gam_shape);
        X = repmat(Std,T,1) .* randn(T,ndim);
    case 'diag'
        for k=1:K
            Std = sqrt(hmm.state(k).Omega.Gam_rate / hmm.state(k).Omega.Gam_shape);
            X = X + repmat(Gamma(:,k),1,ndim) .* repmat(Std,T,1) .* randn(T,ndim);
        end
    case 'uniquefull'
        Cov = hmm.Omega.Gam_rate / hmm.Omega.Gam_shape;
        X = mvnrnd(mu,Cov);
    case 'full'
        for k=1:K
            Cov = hmm.state(k).Omega.Gam_rate / hmm.state(k).Omega.Gam_shape;
            X = X + repmat(Gamma(:,k),1,ndim) .* mvnrnd(mu,Cov);
        end        
end

end


