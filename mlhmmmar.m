function [hmm,pred,gcovm] = mlhmmmar (X,T,hmm,Gamma,completelags)
% Given the state time courses estimation, does a last estimation of each MAR using (local) ML
%
% INPUT
% X             observations
% T             length of series
% hmm           HMM-MAR structure
% Gamma         p(state given X) - has to be fully defined
% completelags  if 1, the lags are made linear with timelag=1 (i.e. a complete set)

%
% OUTPUT
% hmm           HMM-MAR structure with the coefficients and covariance matrices updated to
%                   follow a maximum-likelihood estimation
% pred          predicted response
% gcovm         covariance matrix of the error for the entire model
%
% Author: Diego Vidaurre, OHBA, University of Oxford

if nargin<5, completelags = 0; end
    
K = size(Gamma,2);
ndim = size(X,2);
train = hmm.train;

Sind = train.Sind; S = train.S==1; regressed = sum(S,1)>0;
if ~train.zeromean, Sind = [true(1,size(X,2)); Sind]; end
residuals =  getresiduals(X,T,Sind,train.maxorder,train.order,train.orderoffset,train.timelag,train.exptimelag,train.zeromean);
pred = zeros(size(residuals));
if completelags
    hmm.train.orderoffset=0; hmm.train.timelag=1; hmm.train.exptimelag=0;
    if hmm.train.multipleConf
        for k=1:K
            if isfield(hmm.state(k),'train') && ~isempty(hmm.state(k).train),
                hmm.state(k).train.orderoffset=0; 
                hmm.state(k).train.timelag=1; 
                hmm.state(k).train.exptimelag=0;
            end
        end
    end
end
setxx; % build XX 

for k=1:K
    setstateoptions;
    if isfield(hmm.state(k).W,'S_W'), hmm.state(k).W = rmfield(hmm.state(k).W,'S_W'); end
    if train.uniqueAR
        XY = zeros(size(XX{kk},1)*ndim,1);
        XGX = zeros(size(XX{kk},2)/ndim,size(XX{kk},2)/ndim);
        for n=1:ndim
            ind = n:ndim:size(XX{kk},2);
            iomegan = omega.Gam_shape / omega.Gam_rate(n);
            XGX = XGX + iomegan * XXGXX{k}(ind,ind);
            XY = XY + (iomegan * XX{kk}(:,ind)' .* repmat(Gamma(:,k)',sum(ind),1)) * residuals(:,n);
        end
        hmm.state(k).W.Mu_W = XGX \ XY;
        predk = XX{kk} * repmat(hmm.state(k).W.Mu_W,1,ndim);
    elseif all(S(:)==1)
        hmm.state(k).W.Mu_W = pinv(XX{kk} .* repmat(sqrt(Gamma(:,k)),1,size(XX{kk},2))) * residuals;
        predk = XX{kk} * hmm.state(k).W.Mu_W;
    else
        hmm.state(k).W.Mu_W = zeros(size(hmm.state(k).W.Mu_W));
        for n=1:ndim
            if ~regressed(n), continue; end
            hmm.state(k).W.Mu_W(Sind(:,n),n) = pinv(XX{kk}(:,Sind(:,n)) .* repmat(sqrt(Gamma(:,k)),1,sum(Sind(:,n)))) * residuals(:,n);
        end
        predk = XX{kk} * hmm.state(k).W.Mu_W;
    end
    
    pred = pred + repmat(Gamma(:,k),1,ndim) .* predk;
    e = residuals(:,regressed) - predk(:,regressed);
    if strcmp(train.covtype,'diag')
        hmm.state(k).Omega.Gam_rate(regressed) = 0.5* sum( repmat(Gamma(:,k),1,sum(regressed)) .* e.^2 );
    elseif strcmp(train.covtype,'full')
        hmm.state(k).Omega.Gam_rate(regressed,regressed) =  (e' .* repmat(Gamma(:,k)',sum(regressed),1)) * e;
        hmm.state(k).Omega.Gam_irate(regressed,regressed) = inv(hmm.state(k).Omega.Gam_rate(regressed,regressed));
    end
end
ge = residuals(:,regressed) - pred(:,regressed);
if strcmp(hmm.train.covtype,'uniquediag')
    hmm.Omega.Gam_rate(regressed) = 0.5* sum( ge.^2 );
elseif strcmp(hmm.train.covtype,'uniquefull')
    hmm.Omega.Gam_rate(regressed,regressed) =  (ge' * ge);
    hmm.Omega.Gam_irate(regressed,regressed) = inv(hmm.Omega.Gam_rate(regressed,regressed));
end
gcovm = (ge' * ge) / size(residuals,1);

