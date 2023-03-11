function [Gamma,Gammasum,Xi,LL] = hsinference(data,T,hmm,residuals,options,XX,Gamma0,Xi0,cv)
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
% Gamma0    initial Gamma (only used for ehmm model)
% cv        is this called from a CV routine? 
%
% OUTPUT
%
% Gamma     Probability of hidden state given the data
% Gammasum  sum of Gamma over t
% Xi        joint Prob. of child and parent states given the data
% LL        Log-likelihood, summed across time points
%
% Author: Diego Vidaurre, OHBA, University of Oxford

if length(hmm.train.embeddedlags_batched) > 1
    [Gamma,Gammasum,Xi,LL] = hsinference_batched(T,hmm,residuals,XX);
    return
end
        
N = length(T);
K = hmm.K;
p = hmm.train.lowrank; do_HMM_pca = (p > 0);
if nargin < 7, Gamma0 = []; end
if nargin < 8, Xi0 = []; end
if nargin < 9, cv = 0; end
if ~isfield(hmm,'train')
    if nargin<5 || isempty(options)
        error('You must specify the field options if hmm.train is missing');
    end
    hmm.train = checkoptions(options,data.X,T,0);
end
mixture_model = isfield(hmm.train,'id_mixture') && hmm.train.id_mixture;
if isfield(hmm.train,'episodic'), episodic = hmm.train.episodic; 
else, episodic = 0; 
end
if episodic, rangeK = 1:K+1; else, rangeK = 1:K; end
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
        hmm.train.Sind = formindexes(orders,hmm.train.S) == 1;
        if ~hmm.train.zeromean, hmm.train.Sind = [true(1,ndim); hmm.train.Sind]; end
    end
    residuals =  getresiduals(data.X,T,hmm.train.S,hmm.train.maxorder,hmm.train.order,...
        hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag,hmm.train.zeromean);
else
    ndim = size(residuals,2);
end

if (episodic && ~isfield(hmm.state(1),'P')) || (~episodic && ~isfield(hmm,'P'))
    hmm = hmmhsinit(hmm);
end

if nargin<6 || isempty(XX)
    setxx;
end

if ~hmm.train.acrosstrial_constrained
    Gamma = cell(N,1);
    LL = cell(N,1);
end
Gammasum = zeros(N,K);
if ~mixture_model
    Xi = cell(N,1);
else
    Xi = [];
end

setstateoptions;
% Cache shared results for use in obslike
for k = rangeK
    %hmm.cache = struct();
    hmm.cache.train{k} = train;
    %hmm.cache.order{k} = order;
    %hmm.cache.orders{k} = orders;
    %hmm.cache.Sind{k} = Sind;
    %hmm.cache.S{k} = S;
    if do_HMM_pca
        W = hmm.state(k).W.Mu_W;
        v = hmm.Omega.Gam_rate / hmm.Omega.Gam_shape;
        C = W * W' + v * eye(ndim); 
        ldetWishB = 0.5*logdet(C); PsiWish_alphasum = 0;
    elseif k == 1 && strcmp(train.covtype,'uniquediag') || strcmp(train.covtype,'shareddiag') 
        ldetWishB = 0;
        PsiWish_alphasum = 0;
        for n = 1:ndim
            if ~regressed(n), continue; end
            ldetWishB = ldetWishB+0.5*log(hmm.Omega.Gam_rate(n));
            PsiWish_alphasum = PsiWish_alphasum+0.5*psi(hmm.Omega.Gam_shape);
        end
        C = hmm.Omega.Gam_shape ./ hmm.Omega.Gam_rate;
    elseif k == 1 && strcmp(train.covtype,'uniquefull') || strcmp(train.covtype,'sharedfull')
        ldetWishB = 0.5*logdet(hmm.Omega.Gam_rate(regressed,regressed));
        PsiWish_alphasum = 0;
        for n = 1:sum(regressed)
            PsiWish_alphasum = PsiWish_alphasum+psi(hmm.Omega.Gam_shape/2+0.5-n/2);
        end
        PsiWish_alphasum=PsiWish_alphasum*0.5;
        C = hmm.Omega.Gam_shape * hmm.Omega.Gam_irate;
        for kk = rangeK
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
        if episodic, iC = hmm.state(k).Omega.Gam_rate / hmm.state(k).Omega.Gam_shape; end
    elseif strcmp(train.covtype,'full')
        ldetWishB = 0.5*logdet(hmm.state(k).Omega.Gam_rate(regressed,regressed));
        PsiWish_alphasum = 0;
        for n = 1:sum(regressed)
            PsiWish_alphasum = PsiWish_alphasum+0.5*psi(hmm.state(k).Omega.Gam_shape/2+0.5-n/2);
        end
        C = hmm.state(k).Omega.Gam_shape * hmm.state(k).Omega.Gam_irate;
        if episodic, iC = hmm.state(k).Omega.Gam_rate / hmm.state(k).Omega.Gam_shape; end
        [hmm.cache.WishTrace{k},hmm.cache.codevals] = computeWishTrace(hmm,regressed,XX,C,k);
    end
    % Set up cache
    if ~isfield(train,'distribution') || strcmp(train.distribution,'Gaussian')
        hmm.cache.ldetWishB{k} = ldetWishB;
        hmm.cache.PsiWish_alphasum{k} = PsiWish_alphasum;
        hmm.cache.C{k} = C;
        if episodic && (strcmp(train.covtype,'diag') || strcmp(train.covtype,'full')) 
            hmm.cache.iC{k} = iC; 
        end
    end
end

if hmm.train.acrosstrial_constrained % state probabilities are shared across trials
    
    ttrial = T(1)-order; L = zeros(N*ttrial,K);
    if episodic
        [Gammastar,Xistar,LL] = fb_Gamma_inference_ehmm(XX,hmm,residuals,Gamma0,Xi0,T);
    else
        for j = 1:N
            t = (1:ttrial) + (j-1)*ttrial;
            t2 = (1:ttrial+order) + (j-1)*(ttrial+order);
            try
                L(t2,:) = obslike([],hmm,residuals(t,:),XX(t,:),hmm.cache,cv);
            catch
                error('obslike function is giving trouble - Out of precision?')
            end
        end
        L = squeeze(sum(reshape(L,[ttrial+order,N,K]),2));
        if mixture_model
            Gammastar = id_Gamma_inference(L,hmm.Pi,order); Xistar = [];
        else
            [Gammastar,Xistar,LL] = fb_Gamma_inference_sub(L,hmm.P,hmm.Pi,T(1),...
                order,hmm.train.Gamma_constraint);
        end
    end
    Gamma = zeros(ttrial,N,K); 
    if episodic
        Xi = zeros(ttrial-1,N,K,4);
    elseif mixture_model
        Xi = zeros(ttrial-1,N,K,K);
    end
    for t = 1:ttrial
        Gamma(t,:,:) = repmat(Gammastar(t,:),N,1); 
        if t==ttrial, break; end
        if episodic
            Xi(t,:,:,:) = reshape(repmat(Xistar(t,:,:),N,1),[1,N,K,4]);
        elseif mixture_model
            Xi(t,:,:,:) = reshape(repmat(Xistar(t,:,:),N,1),[1,N,K,K]);
        end
    end
    Gamma = reshape(Gamma,ttrial*N,K);
    if episodic
        Xi = reshape(Xi,(ttrial-1)*N,K,2,2);
    elseif mixture_model
        Xi = reshape(Xi,(ttrial-1)*N,K,K);
    end

elseif hmm.train.useParallel==1 && N>1
    
    % to duplicate this code is really ugly but there doesn't seem to be
    % any other way - more Matlab's fault than mine

    % to avoid the entire data being copied entirely in each parfor loop
    XX_copy = cell(N,1); residuals_copy = cell(N,1); 
    Gamma0_copy = cell(N,1); Xi0_copy = cell(N,1);
    C_copy = cell(N,1);
    for j = 1:N
        t0 = sum(T(1:j-1)); s0 = t0 - order*(j-1); s02 = t0 - (order+1)*(j-1);
        XX_copy{j} = XX(s0+1:s0+T(j)-order,:);
        if ~do_HMM_pca
            residuals_copy{j} = residuals(s0+1:s0+T(j)-order,:);
        end
        if isfield(data,'C')
            C_copy{j} = data.C(s0+1:s0+T(j)-order,:);
        end
        if ~isempty(Gamma0)
            Gamma0_copy{j} = Gamma0(s0+1:s0+T(j)-order,:);
        end
        if ~isempty(Xi0)
            Xi0_copy{j} = Xi(s02+1:s02+T(j)-order-1,:,:,:);
        end        
    end; clear data; clear Gamma0; clear Xi0; clear XX; clear residuals
    
    parfor j = 1:N
        xit = []; llt = [];
        if order>0
            R = [zeros(order,size(residuals_copy{j},2)); residuals_copy{j}];
            if ~isempty(C_copy{j})
                C = [zeros(order,K); C_copy{j}];
            else
                C = NaN(size(R,1),K);
            end
        else
            if do_HMM_pca
                R = XX_copy{j};
            else
                R = residuals_copy{j};
            end
            if ~isempty(C_copy{j})
                C = C_copy{j};
            else
                C = NaN(size(R,1),K);
            end
        end
        % we jump over the fixed parts of the chain
        t = order+1;
        nsegments = computeNoSegments(T(j),t,C);
        xi = cell(nsegments,1); gamma = cell(nsegments,1);
        ll = cell(nsegments,1); ns = 1;
        while t <= T(j)
            if isnan(C(t,1)), no_c = find(~isnan(C(t:T(j),1)));
            else, no_c = find(isnan(C(t:T(j),1)));
            end
            if t>order+1
                if isempty(no_c), slicer = (t-1):T(j); %slice = (t-order-1):T(in);
                else, slicer = (t-1):(no_c(1)+t-2); %slice = (t-order-1):(no_c(1)+t-2);
                end
            else
                if isempty(no_c), slicer = t:T(j); %slice = (t-order):T(in);
                else, slicer = t:(no_c(1)+t-2); %slice = (t-order):(no_c(1)+t-2);
                end
            end
            slicepoints = slicer - order;
            XXt = XX_copy{j}(slicepoints,:); 
            if isnan(C(t,1))
                if episodic
                    slicer2 = slicer(1:end-1);
                    Gamma0t = Gamma0_copy{j}(slicer,:);
                    Xi0t = []; if ~isempty(Xi0_copy{j}), Xi0t = Xi0_copy{j}(slicer2,:); end
                    [gammat,xit,llt] = fb_Gamma_inference_ehmm(XXt,hmm,R(slicer,:),Gamma0t,Xi0t);
                else
                    [gammat,xit,llt] = ...
                        fb_Gamma_inference(XXt,hmm,R(slicer,:),slicepoints,hmm.train.Gamma_constraint,cv);
                end
            else
                gammat = zeros(length(slicer),K);
                if t==order+1, gammat(1,:) = C(slicer(1),:); end
                if ~mixture_model
                    if episodic, xit = zeros(length(slicer)-1, K, 4);
                    else, xit = zeros(length(slicer)-1, K^2);
                    end
                end
                for i = 2:length(slicer)
                    gammat(i,:) = C(slicer(i),:);
                    if ~mixture_model
                        if episodic
                            for k = 1:K
                                gg = [gammat(i-1,k) (1-gammat(i-1,k))];
                                xitr = gg' * gg; 
                                xit(i-1,k,:) = xitr(:)';
                            end
                        else
                            xitr = gammat(i-1,:)' * gammat(i,:) ;
                            xit(i-1,:) = xitr(:)';
                        end
                    end
                end
            end
            if t>order+1
                gammat = gammat(2:end,:);
            end
            if ~mixture_model
                xi{ns} = xit;
            end
            gamma{ns} = gammat;
            ll{ns} = llt;
            ns = ns + 1;
            if isempty(no_c), break;
            else, t = no_c(1)+t-1;
            end
        end
        Gamma{j} = cell2mat(gamma);
        Gammasum(j,:) = sum(Gamma{j});
        LL{j} = cell2mat(ll);
        if ~mixture_model
            if episodic
                Xi{j} = reshape(cell2mat(xi),T(j)-order-1,K,2,2);
            else
                Xi{j} = reshape(cell2mat(xi),T(j)-order-1,K,K);
            end
        end
    end
    
else
 
    for j = 1:N % this is exactly the same than the code above but changing parfor by for
 
        t0 = sum(T(1:j-1)); s0 = t0 - order*(j-1); s02 = t0 - (order+1)*(j-1);
        if order > 0
            R = [zeros(order,size(residuals,2)); residuals(s0+1:s0+T(j)-order,:)];
            if isfield(data,'C')
                C = [zeros(order,K); data.C(s0+1:s0+T(j)-order,:)];
            else
                C = NaN(size(R,1),K);
            end
        else %s0==t0
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
        nsegments = computeNoSegments(T(j),t,C);
        xi = cell(nsegments,1); gamma = cell(nsegments,1);
        ll = cell(nsegments,1); ns = 1; 
        while t <= T(j)
            if isnan(C(t,1)), no_c = find(~isnan(C(t:T(j),1)));
            else, no_c = find(isnan(C(t:T(j),1)));
            end
            if t > order+1
                if isempty(no_c), slicer = (t-1):T(j); %slice = (t-order-1):T(in);
                else, slicer = (t-1):(no_c(1)+t-2); %slice = (t-order-1):(no_c(1)+t-2);
                end
            else
                if isempty(no_c), slicer = t:T(j); %slice = (t-order):T(in);
                else, slicer = t:(no_c(1)+t-2); %slice = (t-order):(no_c(1)+t-2);
                end
            end
            slicepoints = slicer + s0 - order;
            XXt = XX(slicepoints,:);
            if isnan(C(t,1))
                if episodic
                    slicepoints2 = slicer(1:end-1) + s02 - order;
                    Gamma0t = Gamma0(slicepoints,:);
                    Xi0t = []; if ~isempty(Xi0), Xi0t = Xi0(slicepoints2,:,:,:); end
                    [gammat,xit,llt] = ...
                        fb_Gamma_inference_ehmm(XXt,hmm,R(slicer,:),Gamma0t,Xi0t);
                else
                    [gammat,xit,llt] = ...
                        fb_Gamma_inference(XXt,hmm,R(slicer,:),slicepoints,hmm.train.Gamma_constraint,cv);
                end
                if any(isnan(gammat(:))) % this will never come up - we treat it within fb_Gamma_inference
                    error(['State time course inference returned NaN (out of precision). ' ...
                        'There are probably extreme events in the data'])
                end
            else
                gammat = zeros(length(slicer),K);
                llt = NaN(length(slicer),1);
                if t==order+1, gammat(1,:) = C(slicer(1),:); end
                if ~mixture_model
                    if episodic, xit = zeros(length(slicer)-1, K, 4);
                    else, xit = zeros(length(slicer)-1, K^2);
                    end
                end
                for i = 2:length(slicer)
                    gammat(i,:) = C(slicer(i),:);
                    if ~mixture_model
                        if episodic
                            for k = 1:K
                                gg = [gammat(i-1,k) (1-gammat(i-1,k))];
                                xitr = gg' * gg;
                                xit(i-1,k,:) = xitr(:)';
                            end
                        else
                            xitr = gammat(i-1,:)' * gammat(i,:) ;
                            xit(i-1,:) = xitr(:)';
                        end
                    end
                end
            end
            if t > order+1
                gammat = gammat(2:end,:);
            end
            if ~mixture_model
                xi{ns} = xit; 
            end
            gamma{ns} = gammat;
            ll{ns} = llt;
            ns = ns + 1; 
            if isempty(no_c), break;
            else, t = no_c(1)+t-1;
            end
        end
        Gamma{j} = cell2mat(gamma);
        Gammasum(j,:) = sum(Gamma{j});
        LL{j} = cell2mat(ll);
        if ~mixture_model
            if episodic
                Xi{j} = reshape(cell2mat(xi),T(j)-order-1,K,2,2);
            else
                Xi{j} = reshape(cell2mat(xi),T(j)-order-1,K,K);
            end
        end
    end
    
end

% join
if ~hmm.train.acrosstrial_constrained
    Gamma = cell2mat(Gamma);
    LL = cell2mat(LL);
    if ~mixture_model, Xi = cell2mat(Xi); end
end

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


function nsegments = computeNoSegments(Tj,t0,C)
t = t0; nsegments = 0;
while t <= Tj
    if isnan(C(t,1)), no_c = find(~isnan(C(t:Tj,1)));
    else, no_c = find(isnan(C(t:Tj,1)));
    end
    nsegments = nsegments + 1;
    if isempty(no_c), break;
    else, t = no_c(1)+t-1;
    end
end
end