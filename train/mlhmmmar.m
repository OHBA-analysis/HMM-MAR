function hmm = mlhmmmar (X,T,hmm0,Gamma,completelags)
% Given the state time courses estimation, does a last estimation of each MAR using (local) ML
%
% INPUT
% X             observations
% T             length of series
% hmm0          HMM-MAR structure
% Gamma         p(state given X) - has to be fully defined
% completelags  if 1, the lags are made linear with timelag=1 (i.e. a complete set)

%
% OUTPUT
% hmm           HMM-MAR structure with the coefficients and covariance matrices updated to
%                   follow a maximum-likelihood estimation
% gcovm         covariance matrix of the error for the entire model
%
% Author: Diego Vidaurre, OHBA, University of Oxford
 
hmm = hmm0;

if hmm.train.uniqueAR
    error('mlhmmmar not yet implemented for uniqueAR')
end

if iscell(T)
    T2 = cell2mat(T); T2 = T2(:);
else
    T2 = T;
end
if nargin<4, Gamma = ones(sumT-length(T2)*hmm.train.order,1); end
if nargin<5, completelags = 0; end
    
K = size(Gamma,2);
hmm.K = K; N = length(T);

if completelags
    maxorder0 = hmm.train.maxorder;
    hmm.train.orderoffset=0; hmm.train.timelag=1; hmm.train.exptimelag=0;
    hmm.train.maxorder = hmm.train.order;
    maxorderd = hmm.train.maxorder - maxorder0;
    if maxorderd>0 % trim Gamma if maxorder has changed
        Gamma0 = Gamma;
        Gamma = zeros(sum(T)-hmm.train.maxorder*length(T),K);
        for j = 1:N
            t00 = sum(T(1:j-1)) - (j-1)*maxorder0 + 1;
            t10 = sum(T(1:j)) - j*maxorder0;
            t0 = sum(T(1:j-1)) - (j-1)*hmm.train.maxorder + 1;
            t1 = sum(T(1:j)) - j*hmm.train.maxorder;
            Gamma(t0:t1,:) = Gamma0(t00+maxorderd:t10,:);
        end
    end
end

[hmm.train.orders,order] = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);

for k=1:K
    if ~isfield(hmm.state(k),'train') || isempty(hmm.state(k).train)
        hmm.state(k).train = hmm.train;
    end
end

if isfield(hmm.train,'B'), hmm.train.B = []; end
if isfield(hmm.train,'V'), hmm.train.V = []; end

if iscell(X)
    c = 0;
    for i = 1:N
        [~,XX,Y] = loadfile(X{i},T{i},hmm.train);
        if i==1, ndim = size(Y,2); end
        ind = (1:sum(T{i})-length(T{i})*order) + c;
        c = c + length(ind);
        %else
        %    ind = sum(T(1:i-1)) + (1:T(i));
        %    Y = getresiduals(data(ind,:),T(i),Sind,hmm.train.maxorder,hmm.train.order,hmm.train.orderoffset,...
        %        hmm.train.timelag,hmm.train.exptimelag,hmm.train.zeromean);
        %    XX = formautoregr(data.X,T,orders,hmm.train.maxorder,hmm.train.zeromean,0,B,V);
        %end
        if i==1
            XX2 = zeros(size(XX,2),size(XX,2),K);
            XXY = zeros(size(XX,2),ndim,K);
        end
        for k=1:K
            XX2(:,:,k) = XX2(:,:,k) + (XX' .* repmat(Gamma(ind,k)',size(XX,2),1)) * XX;
            XXY(:,:,k) = XXY(:,:,k) + (XX' .* repmat(Gamma(ind,k)',size(XX,2),1)) * Y;
        end
    end
    for k=1:K
        hmm.state(k).W.Mu_W = (XX2(:,:,k) + 1e-6 * eye(size(XX2,2))) \ XXY(:,:,k);
        %hmm.state(k).W.Mu_W = XX2(:,:,k) \ XXY(:,:,k);
        if isfield(hmm.state(k).W,'S_W'), hmm.state(k).W = rmfield(hmm.state(k).W,'S_W'); end
        if isfield(hmm.state(k).W,'iS_W'), hmm.state(k).W = rmfield(hmm.state(k).W,'iS_W'); end
    end
    %sumT = 0; ge = zeros(ndim);
    c = 0;
    for i = 1:N
        [~,XX,Y] = loadfile(X{i},T{i},hmm.train);
        ind = (1:sum(T{i})-length(T{i})*order) + c;
        c = c + length(ind);
        %sumT = sumT + size(Y,1); 
        if i==1
            if strcmp(hmm.train.covtype,'uniquediag')
                hmm.Omega.Gam_rate = zeros(1,ndim); 
                hmm.Omega.Gam_shape = 0; 
            elseif strcmp(hmm.train.covtype,'uniquefull')
                hmm.Omega.Gam_rate = zeros(ndim); 
                hmm.Omega.Gam_shape = 0; 
            elseif strcmp(hmm.train.covtype,'diag')
                for k=1:K
                    hmm.state(k).Omega.Gam_rate = zeros(1,ndim);
                    hmm.state(k).Omega.Gam_shape = 0;
                end
            else % full
                for k=1:K
                    hmm.state(k).Omega.Gam_rate = zeros(ndim);
                    hmm.state(k).Omega.Gam_shape = 0;
                end
            end
        end
        for k=1:K
            e = Y - XX * hmm.state(k).W.Mu_W;
            %ge = ge + e' * e;
            if strcmp(hmm.train.covtype,'diag')
                hmm.state(k).Omega.Gam_shape = hmm.state(k).Omega.Gam_shape + 0.5 * sum(Gamma(ind,k));
                hmm.state(k).Omega.Gam_rate = hmm.state(k).Omega.Gam_rate + ...
                    0.5 *  sum( repmat(Gamma(ind,k),1,ndim) .* e.^2 );
            elseif strcmp(hmm.train.covtype,'full')
                hmm.state(k).Omega.Gam_shape = hmm.state(k).Omega.Gam_shape + sum(Gamma(ind,k));
                hmm.state(k).Omega.Gam_rate = hmm.state(k).Omega.Gam_rate + ...
                    (e' .* repmat(Gamma(ind,k)',ndim,1)) * e;
            elseif strcmp(hmm.train.covtype,'uniquediag')
                hmm.Omega.Gam_shape = hmm.Omega.Gam_shape + 0.5 * sum(Gamma(ind,k));
                hmm.Omega.Gam_rate = hmm.Omega.Gam_rate + 0.5 *  sum( repmat(Gamma(ind,k),1,ndim) .* e.^2 );
            else
                hmm.Omega.Gam_shape = hmm.Omega.Gam_shape + sum(Gamma(ind,k));
                hmm.Omega.Gam_rate = hmm.Omega.Gam_rate + (e' .* repmat(Gamma(ind,k)',ndim,1)) * e;
            end
        end
 
    end
    
else
    
    ndim = size(X,2);
    S = hmm.train.S==1; regressed = sum(S,1)>0;
    Sind = formindexes(hmm.train.orders,hmm.train.S); hmm.train.Sind = Sind;
    if ~hmm.train.zeromean, Sind = [true(1,size(X,2)); Sind]; end
    Y = getresiduals(X,T,Sind,hmm.train.maxorder,hmm.train.order,hmm.train.orderoffset,...
        hmm.train.timelag,hmm.train.exptimelag,hmm.train.zeromean);
    pred = zeros(size(Y));
    setxx; % build XX
    
    for k=1:K
        setstateoptions;        
        if hmm.train.uniqueAR
            XY = zeros(size(XX,1)*ndim,1);
            XGX = zeros(size(XX,2)/ndim,size(XX,2)/ndim);
            for n=1:ndim
                ind = n:ndim:size(XX,2);
                iomegan = omega.Gam_shape / omega.Gam_rate(n);
                XGX = XGX + iomegan * XXGXX{k}(ind,ind);
                XY = XY + (iomegan * XX(:,ind)' .* repmat(Gamma(:,k)',sum(ind),1)) * Y(:,n);
            end
            hmm.state(k).W.Mu_W = XGX \ XY;
            predk = XX * repmat(hmm.state(k).W.Mu_W,1,ndim);
        elseif all(S(:)==1)
            hmm.state(k).W.Mu_W = pinv(XX .* repmat(sqrt(Gamma(:,k)),1,size(XX,2))) ...
                * ( repmat(sqrt(Gamma(:,k)),1,size(Y,2)) .* Y);
            predk = XX * hmm.state(k).W.Mu_W;
        else
            hmm.state(k).W.Mu_W = zeros(size(XX,1),ndim);
            for n=1:ndim
                if ~regressed(n), continue; end
                hmm.state(k).W.Mu_W(Sind(:,n),n) = pinv(XX(:,Sind(:,n)) .* ...
                    repmat(sqrt(Gamma(:,k)),1,sum(Sind(:,n)))) * Y(:,n);
            end
            predk = XX * hmm.state(k).W.Mu_W;
        end
        
        pred = pred + repmat(Gamma(:,k),1,ndim) .* predk;
        e = Y(:,regressed) - predk(:,regressed);
        if strcmp(hmm.train.covtype,'diag')
            hmm.state(k).Omega.Gam_shape = 0.5 * sum(Gamma(:,k));
            hmm.state(k).Omega.Gam_rate = zeros(1,ndim);
            hmm.state(k).Omega.Gam_rate(regressed) = 0.5 * ...
                sum( repmat(Gamma(:,k),1,sum(regressed)) .* e.^2 );
        elseif strcmp(hmm.train.covtype,'full')
            hmm.state(k).Omega.Gam_shape = sum(Gamma(:,k));
            hmm.state(k).Omega.Gam_rate = zeros(ndim);
            hmm.state(k).Omega.Gam_rate(regressed,regressed) =  ...
                (e' .* repmat(Gamma(:,k)',sum(regressed),1)) * e;
            hmm.state(k).Omega.Gam_irate(regressed,regressed) = ...
                inv(hmm.state(k).Omega.Gam_rate(regressed,regressed));
        end
    end
    % if length(hmm.state) > K
    %     state = hmm.state(1:K);
    %     hmm = rmfield(hmm,'state');
    %     hmm.state = state;
    % end
    
    e = Y(:,regressed) - pred(:,regressed);
    if strcmp(hmm.train.covtype,'uniquediag')
       hmm.Omega.Gam_shape = 0.5 * sum(T);
       hmm.Omega.Gam_rate = zeros(1,ndim);
       hmm.Omega.Gam_rate(regressed) = 0.5 * sum( e.^2 );
    elseif strcmp(hmm.train.covtype,'uniquefull')
       hmm.Omega.Gam_shape = sum(T);
       hmm.Omega.Gam_rate = zeros(ndim);
       hmm.Omega.Gam_rate(regressed,regressed) =  (e' * e);
       hmm.Omega.Gam_irate(regressed,regressed) = inv(hmm.Omega.Gam_rate(regressed,regressed));
    end
    %gcovm = (ge' * ge) / size(Y,1);
    
end

end

