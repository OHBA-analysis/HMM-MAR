function [Gamma,Gammasum,Xi,LL,B] = hsinference(data,T,hmm,residuals,options,XX)
%
% inference engine for HMMs.
%
% INPUT
%
% data      Observations - a struct with X (time series) and C (classes)
% T         Number of time points for each time series
% hmm       hmm data structure
% residuals in case we train on residuals, the value of those.
% XX        optionally, XX, as computed by setxx.m, can be supplied
%
% OUTPUT
%
% Gamma     Probability of hidden state given the data
% Gammasum  sum of Gamma over t
% Xi        joint Prob. of child and parent states given the data
% LL        Log-likelihood
%
% Author: Diego Vidaurre, OHBA, University of Oxford

N = length(T);
K = length(hmm.state);
p = hmm.train.lowrank; do_HMM_pca = (p > 0);

mixture_model = isfield(hmm.train,'id_mixture') && hmm.train.id_mixture;

if ~isfield(hmm,'train')
    if nargin<5 || isempty(options)
        error('You must specify the field options if hmm.train is missing');
    end
    hmm.train = checkoptions(options,data.X,T,0);
end
order = hmm.train.maxorder;

if iscell(data)
    data = cell2mat(data);
end
if ~isstruct(data)
    data = struct('X',data);
    data.C = NaN(size(data.X,1)-order*length(T),K);
end

if do_HMM_pca
    ndim = size(data.X,2);
elseif nargin<4 || isempty(residuals)
    ndim = size(data.X,2);
    if ~isfield(hmm.train,'Sind')
        orders = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
        hmm.train.Sind = formindexes(orders,hmm.train.S);
    end
    if ~hmm.train.zeromean, hmm.train.Sind = [true(1,ndim); hmm.train.Sind]; end
    residuals =  getresiduals(data.X,T,hmm.train.Sind,hmm.train.maxorder,hmm.train.order,...
        hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag,hmm.train.zeromean);
else
    ndim = size(residuals,2);
end

if ~isfield(hmm,'P')
    hmm = hmmhsinit(hmm);
end

if nargin<6 || isempty(XX)
    setxx;
end

Gamma = cell(N,1);
LL = zeros(N,1);
Gammasum = zeros(N,K);
if ~mixture_model
    Xi = cell(N,1);
else
    Xi = [];
end
B = cell(N,1);

n_argout = nargout;
S = hmm.train.S==1;
regressed = sum(S,1)>0;
setstateoptions;
% Cache shared results for use in obslike
for k = 1:K
    
    %hmm.cache = struct();
    hmm.cache.train{k} = train;
    hmm.cache.order{k} = order;
    hmm.cache.orders{k} = orders;
    hmm.cache.Sind{k} = Sind;
    hmm.cache.S{k} = S;
   
    if do_HMM_pca
        W = hmm.state(k).W.Mu_W;
        v = hmm.Omega.Gam_rate / hmm.Omega.Gam_shape;
        C = W * W' + v * eye(ndim); 
        ldetWishB = 0.5*logdet(C); PsiWish_alphasum = 0;
    elseif k == 1 && strcmp(train.covtype,'uniquediag') 
        ldetWishB = 0;
        PsiWish_alphasum = 0;
        for n = 1:ndim
            if ~regressed(n), continue; end
            ldetWishB = ldetWishB+0.5*log(hmm.Omega.Gam_rate(n));
            PsiWish_alphasum = PsiWish_alphasum+0.5*psi(hmm.Omega.Gam_shape);
        end
        C = hmm.Omega.Gam_shape ./ hmm.Omega.Gam_rate;
    elseif k == 1 && strcmp(train.covtype,'uniquefull')
        ldetWishB = 0.5*logdet(hmm.Omega.Gam_rate(regressed,regressed));
        PsiWish_alphasum = 0;
        for n = 1:sum(regressed)
            PsiWish_alphasum = PsiWish_alphasum+psi(hmm.Omega.Gam_shape/2+0.5-n/2);
        end
        PsiWish_alphasum=PsiWish_alphasum*0.5;
        C = hmm.Omega.Gam_shape * hmm.Omega.Gam_irate;
        for kk = 1:hmm.K
            [hmm.cache.WishTrace{kk},hmm.cache.codevals] = computeWishTrace(hmm,regressed,XX,C,kk);
        end
    elseif strcmp(train.covtype,'diag')
        ldetWishB=0;
        PsiWish_alphasum = 0;
        for n = 1:ndim
            if ~regressed(n), continue; end
            ldetWishB = ldetWishB+0.5*log(hmm.state(k).Omega.Gam_rate(n));
            PsiWish_alphasum = PsiWish_alphasum+0.5*psi(hmm.state(k).Omega.Gam_shape);
        end
        C = hmm.state(k).Omega.Gam_shape ./ hmm.state(k).Omega.Gam_rate;
    elseif strcmp(train.covtype,'full')
        ldetWishB = 0.5*logdet(hmm.state(k).Omega.Gam_rate(regressed,regressed));
        PsiWish_alphasum = 0;
        for n = 1:sum(regressed)
            PsiWish_alphasum = PsiWish_alphasum+0.5*psi(hmm.state(k).Omega.Gam_shape/2+0.5-n/2);
        end
        C = hmm.state(k).Omega.Gam_shape * hmm.state(k).Omega.Gam_irate;
        [hmm.cache.WishTrace{k},hmm.cache.codevals] = computeWishTrace(hmm,regressed,XX,C,k);
    end
    if  (~isfield(train,'distribution') || ~strcmp(train.distribution,'logistic'))
        hmm.cache.ldetWishB{k} = ldetWishB;
        hmm.cache.PsiWish_alphasum{k} = PsiWish_alphasum;
        hmm.cache.C{k} = C;
    end
end

if hmm.train.useParallel==1 && N>1
    
    % to duplicate this code is really ugly but there doesn't seem to be
    % any other way - more Matlab's fault than mine
    parfor j = 1:N
        xit = [];
        Bt = [];
        t0 = sum(T(1:j-1)); s0 = t0 - order*(j-1);
        if order>0
            R = [zeros(order,size(residuals,2));  residuals(s0+1:s0+T(j)-order,:)];
            if isfield(data,'C')
                C = [zeros(order,K); data.C(s0+1:s0+T(j)-order,:)];
            else
                C = NaN(size(R,1),K);
            end
        else
            if do_HMM_pca
                R = XX(s0+1:s0+T(j),:);
            else
                R = residuals(s0+1:s0+T(j),:);
            end
            if isfield(data,'C')
                C = data.C(s0+1:s0+T(j),:);
            else
                C = NaN(size(R,1),K);
            end
        end
        % we jump over the fixed parts of the chain
        t = order+1;
        xi = []; gamma = []; gammasum = zeros(1,K); ll = 0;
        while t <= T(j)
            if isnan(C(t,1)), no_c = find(~isnan(C(t:T(j),1)));
            else no_c = find(isnan(C(t:T(j),1)));
            end
            if t>order+1
                if isempty(no_c), slicer = (t-1):T(j); %slice = (t-order-1):T(in);
                else slicer = (t-1):(no_c(1)+t-2); %slice = (t-order-1):(no_c(1)+t-2);
                end
            else
                if isempty(no_c), slicer = t:T(j); %slice = (t-order):T(in);
                else slicer = t:(no_c(1)+t-2); %slice = (t-order):(no_c(1)+t-2);
                end
            end
            slicepoints = slicer + s0 - order;
            XXt = XX(slicepoints,:); 
            if isnan(C(t,1))
                [gammat,xit,Bt] = ...
                    fb_Gamma_inference(XXt,K,hmm,R(slicer,:),slicepoints,hmm.train.Gamma_constraint);
            else
                gammat = zeros(length(slicer),K);
                if t==order+1, gammat(1,:) = C(slicer(1),:); end
                if ~mixture_model, xit = zeros(length(slicer)-1, K^2); end
                for i = 2:length(slicer)
                    gammat(i,:) = C(slicer(i),:);
                    if ~mixture_model
                        xitr = gammat(i-1,:)' * gammat(i,:) ;
                        xit(i-1,:) = xitr(:)';
                    end
                end
                if n_argout>=4, Bt = obslike([],hmm,R(slicer,:),XXt,hmm.cache); end
            end
            if t>order+1
                gammat = gammat(2:end,:);
            end
            if ~mixture_model, xi = [xi; xit]; end
            gamma = [gamma; gammat];
            gammasum = gammasum + sum(gamma);
            if n_argout>=4 
                ll = ll + sum(log(sum(Bt(order+1:end,:) .* gammat, 2))); 
            end
            if n_argout>=5, B{j} = [B{j}; Bt(order+1:end,:) ]; end
            if isempty(no_c), break;
            else, t = no_c(1)+t-1;
            end
        end
        Gamma{j} = gamma;
        Gammasum(j,:) = gammasum;
        if n_argout>=4, LL(j) = ll; end
        %Xi=cat(1,Xi,reshape(xi,T(in)-order-1,K,K));
        if ~mixture_model, Xi{j} = reshape(xi,T(j)-order-1,K,K); end
    end
    
else
    
    for j = 1:N % this is exactly the same than the code above but changing parfor by for
        Bt = [];  
        t0 = sum(T(1:j-1)); s0 = t0 - order*(j-1);
        if order>0
            R = [zeros(order,size(residuals,2)); residuals(s0+1:s0+T(j)-order,:)];
            if isfield(data,'C')
                C = [zeros(order,K); data.C(s0+1:s0+T(j)-order,:)];
            else
                C = NaN(size(R,1),K);
            end
        else
            if do_HMM_pca
                R = XX(s0+1:s0+T(j),:);
            else
                R = residuals(s0+1:s0+T(j),:);
            end
            if isfield(data,'C')
                C = data.C(s0+1:s0+T(j),:);
            else
                C = NaN(size(R,1),K);
            end
        end
        % we jump over the fixed parts of the chain
        t = order+1;
        xi = []; gamma = []; gammasum = zeros(1,K); ll = 0;
        while t <= T(j)
            if isnan(C(t,1)), no_c = find(~isnan(C(t:T(j),1)));
            else no_c = find(isnan(C(t:T(j),1)));
            end
            if t > order+1
                if isempty(no_c), slicer = (t-1):T(j); %slice = (t-order-1):T(in);
                else slicer = (t-1):(no_c(1)+t-2); %slice = (t-order-1):(no_c(1)+t-2);
                end
            else
                if isempty(no_c), slicer = t:T(j); %slice = (t-order):T(in);
                else slicer = t:(no_c(1)+t-2); %slice = (t-order):(no_c(1)+t-2);
                end
            end
            slicepoints=slicer + s0 - order;
            XXt = XX(slicepoints,:);
            if isnan(C(t,1))
                [gammat,xit,Bt] = ...
                    fb_Gamma_inference(XXt,K,hmm,R(slicer,:),slicepoints,hmm.train.Gamma_constraint);
                if any(isnan(gammat(:))) % this will never come up - we treat it within fb_Gamma_inference
                    error(['State time course inference returned NaN (out of precision). ' ...
                        'There are probably extreme events in the data'])
                end
            else
                gammat = zeros(length(slicer),K);
                if t==order+1, gammat(1,:) = C(slicer(1),:); end
                if ~mixture_model, xit = zeros(length(slicer)-1, K^2); end
                for i=2:length(slicer)
                    gammat(i,:) = C(slicer(i),:);
                    if ~mixture_model
                        xitr = gammat(i-1,:)' * gammat(i,:) ;
                        xit(i-1,:) = xitr(:)';
                    end
                end
                if nargout>=4, Bt = obslike([],hmm,R(slicer,:),XXt,hmm.cache); end
            end
            if t > order+1
                gammat = gammat(2:end,:);
            end
            if ~mixture_model, xi = [xi; xit]; end
            gamma = [gamma; gammat];
            gammasum = gammasum + sum(gamma);
            if nargout>=4 
                ll = ll + sum(log(sum(Bt(order+1:end,:) .* gammat, 2)));
            end
            if nargout>=5, B{j} = [B{j}; Bt(order+1:end,:) ]; end
            if isempty(no_c), break;
            else t = no_c(1)+t-1;
            end
        end
        Gamma{j} = gamma;
        Gammasum(j,:) = gammasum;
        if nargout>=4, LL(j) = ll; end
        %Xi=cat(1,Xi,reshape(xi,T(in)-order-1,K,K));
        if ~mixture_model, Xi{j} = reshape(xi,T(j)-order-1,K,K); end
    end
end

% join
Gamma = cell2mat(Gamma);
if ~mixture_model, Xi = cell2mat(Xi); end
if n_argout>=5, B  = cell2mat(B); end

% orthogonalise = 1; 
% Gamma0 = Gamma; 
% if orthogonalise && hmm.train.tuda
%     T = length(Gamma) / length(T) * ones(length(T),1); 
%     Gamma = reshape(Gamma,[T(1) length(T) size(Gamma,2) ]);
%     Y = reshape(residuals(:,end),[T(1) length(T)]);
%     for t = 1:T(1)
%        y = Y(t,:)'; 
%        for k = 1:size(Gamma,3)
%            x = Gamma(t,:,k)';
%            b = y \ x;
%            Gamma(t,:,k) = Gamma(t,:,k) - (y * b)';
%        end
%     end
%     Gamma = reshape(Gamma,[T(1)*length(T) size(Gamma,3) ]);
%     Gamma = rdiv(Gamma,sum(Gamma,2));
%     Gamma = Gamma - min(Gamma(:));
%     Gamma = rdiv(Gamma,sum(Gamma,2));
% end

end

function [WishTrace,X_coded_vals] = computeWishTrace(hmm,regressed,XX,C,k)
X = XX(:,~regressed);
if any(hmm.train.S(:)~=1) && length(unique(X))<5
    % regressors are low dim categorical - compute and store in cache for
    % each regressor type - convert to binary code:
    X_coded = X*[2.^(1:size(X,2))]';
    X_coded_vals = unique(X_coded);
    validentries = logical(hmm.train.S(:)==1);
    WishTrace = zeros(1,length(X_coded_vals));
    %for k = 1:hmm.K
        B_S = hmm.state(k).W.S_W(validentries,validentries);
        for i=1:length(X_coded_vals)
            t_samp = find(X_coded==X_coded_vals(i),1);
            WishTrace(i) = trace(kron(C(regressed,regressed),X(t_samp,:)'*X(t_samp,:))*B_S);
        end      
    %end
    
else
    WishTrace =[];
    X_coded_vals=[];
end
end