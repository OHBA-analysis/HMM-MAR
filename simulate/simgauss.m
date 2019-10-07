function X = simgauss(T,hmm,Gamma)

if iscell(T)
    if size(T,1)==1, T = T'; end
    T = cell2mat(T);
end
T = sum(T);

if nargin < 3
    Gamma = ones(sum(T),1);
end

ndim = size(hmm.state(1).W.Mu_W,2); 
if ndim==0, ndim = size(hmm.state(1).Omega.Gam_rate,1); end
K = size(Gamma,2);
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

if ~hmm.train.zeromean
    for k = 1:K
        X = X + repmat(Gamma(:,k),1,ndim) .* ...
            repmat(hmm.state(k).W.Mu_W(1,:),T,1);
    end
end

end


