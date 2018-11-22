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

if nargin<4 || isempty(residuals)
    ndim = size(data.X,2);
    if ~isfield(hmm.train,'Sind')
        orders = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
        hmm.train.Sind = formindexes(orders,hmm.train.S);
    end
    if ~hmm.train.zeromean, hmm.train.Sind = [true(1,ndim); hmm.train.Sind]; end
    residuals =  getresiduals(data.X,T,hmm.train.Sind,hmm.train.maxorder,hmm.train.order,...
        hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag,hmm.train.zeromean);
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
Xi = cell(N,1);
B = cell(N,1);

n_argout = nargout;

ndim = size(residuals,2);
S = hmm.train.S==1;
regressed = sum(S,1)>0;

% Cache shared results for use in obslike
for k = 1:K
    setstateoptions;
    %hmm.cache = struct();
    hmm.cache.train{k} = train;
    hmm.cache.order{k} = order;
    hmm.cache.orders{k} = orders;
    hmm.cache.Sind{k} = Sind;
    hmm.cache.S{k} = S;
    if k == 1 && strcmp(train.covtype,'uniquediag')
        ldetWishB=0;
        PsiWish_alphasum=0;
        for n=1:ndim
            if ~regressed(n), continue; end
            ldetWishB=ldetWishB+0.5*log(hmm.Omega.Gam_rate(n));
            PsiWish_alphasum=PsiWish_alphasum+0.5*psi(hmm.Omega.Gam_shape);
        end
        C = hmm.Omega.Gam_shape ./ hmm.Omega.Gam_rate;
    elseif k == 1 && strcmp(train.covtype,'uniquefull')
        ldetWishB=0.5*logdet(hmm.Omega.Gam_rate(regressed,regressed));
        PsiWish_alphasum=0;
        for n=1:sum(regressed)
            PsiWish_alphasum=PsiWish_alphasum+psi(hmm.Omega.Gam_shape/2+0.5-n/2);
        end
        PsiWish_alphasum=PsiWish_alphasum*0.5;
        C = hmm.Omega.Gam_shape * hmm.Omega.Gam_irate;
    elseif strcmp(train.covtype,'diag')
        ldetWishB=0;
        PsiWish_alphasum=0;
        for n=1:ndim
            if ~regressed(n), continue; end
            ldetWishB=ldetWishB+0.5*log(hmm.state(k).Omega.Gam_rate(n));
            PsiWish_alphasum=PsiWish_alphasum+0.5*psi(hmm.state(k).Omega.Gam_shape);
        end
        C = hmm.state(k).Omega.Gam_shape ./ hmm.state(k).Omega.Gam_rate;
    elseif strcmp(train.covtype,'full')
        ldetWishB=0.5*logdet(hmm.state(k).Omega.Gam_rate(regressed,regressed));
        PsiWish_alphasum=0;
        for n=1:sum(regressed)
            PsiWish_alphasum=PsiWish_alphasum+0.5*psi(hmm.state(k).Omega.Gam_shape/2+0.5-n/2);
        end
        C = hmm.state(k).Omega.Gam_shape * hmm.state(k).Omega.Gam_irate;
    end
    
    hmm.cache.ldetWishB{k} = ldetWishB;
    hmm.cache.PsiWish_alphasum{k} = PsiWish_alphasum;
    hmm.cache.C{k} = C;
    hmm.cache.do_normwishtrace(k) = ~isempty(hmm.state(k).W.Mu_W);
    
end


if hmm.train.useParallel==1 && N>1
    
    % to duplicate this code is really ugly but there doesn't seem to be
    % any other way - more Matlab's fault than mine
    parfor in = 1:N
        Bt = [];  
        t0 = sum(T(1:in-1)); s0 = t0 - order*(in-1);
        if order>0
            C = [zeros(order,K); data.C(s0+1:s0+T(in)-order,:)];
            R = [zeros(order,size(residuals,2));  residuals(s0+1:s0+T(in)-order,:)];
        else
            C = data.C(s0+1:s0+T(in)-order,:);
            R = residuals(s0+1:s0+T(in)-order,:);
        end
        % we jump over the fixed parts of the chain
        t = order+1;
        xi = []; gamma = []; gammasum = zeros(1,K); ll = 0;
        while t<=T(in)
            if isnan(C(t,1)), no_c = find(~isnan(C(t:T(in),1)));
            else no_c = find(isnan(C(t:T(in),1)));
            end
            if t>order+1
                if isempty(no_c), slicer = (t-1):T(in); %slice = (t-order-1):T(in);
                else slicer = (t-1):(no_c(1)+t-2); %slice = (t-order-1):(no_c(1)+t-2);
                end
            else
                if isempty(no_c), slicer = t:T(in); %slice = (t-order):T(in);
                else slicer = t:(no_c(1)+t-2); %slice = (t-order):(no_c(1)+t-2);
                end
            end
            XXt = XX(slicer + s0 - order,:); 
            if isnan(C(t,1))
                [gammat,xit,Bt] = nodecluster(XXt,K,hmm,R(slicer,:),in);
            else
                gammat = zeros(length(slicer),K);
                if t==order+1, gammat(1,:) = C(slicer(1),:); end
                xit = zeros(length(slicer)-1, K^2);
                for i=2:length(slicer)
                    gammat(i,:) = C(slicer(i),:);
                    xitr = gammat(i-1,:)' * gammat(i,:) ;
                    xit(i-1,:) = xitr(:)';
                end
                if n_argout>=4, Bt = obslike([],hmm,R(slicer,:),XXt,hmm.cache); end
            end
            if t>order+1
                gammat = gammat(2:end,:);
            end
            xi = [xi; xit];
            gamma = [gamma; gammat];
            gammasum = gammasum + sum(gamma);
            if n_argout>=4, ll = ll + sum(sum(log(Bt(order+1:end,:)) .* gammat,2)); end
            if n_argout>=5, B{in} = [B{in}; Bt(order+1:end,:) ]; end
            if isempty(no_c), break;
            else, t = no_c(1)+t-1;
            end
        end
        Gamma{in} = gamma;
        Gammasum(in,:) = gammasum;
        if n_argout>=4, LL(in) = ll; end
        %Xi=cat(1,Xi,reshape(xi,T(in)-order-1,K,K));
        Xi{in} = reshape(xi,T(in)-order-1,K,K);
    end
    
else
    
    for in=1:N % this is exactly the same than the code above but changing parfor by for
        Bt = [];  
        t0 = sum(T(1:in-1)); s0 = t0 - order*(in-1);
        if order>0
            C = [zeros(order,K); data.C(s0+1:s0+T(in)-order,:)];
            R = [zeros(order,size(residuals,2));  residuals(s0+1:s0+T(in)-order,:)];
        else
            C = data.C(s0+1:s0+T(in)-order,:);
            R = residuals(s0+1:s0+T(in)-order,:);
        end
        % we jump over the fixed parts of the chain
        t = order+1;
        xi = []; gamma = []; gammasum = zeros(1,K); ll = 0;
        while t<=T(in)
            if isnan(C(t,1)), no_c = find(~isnan(C(t:T(in),1)));
            else no_c = find(isnan(C(t:T(in),1)));
            end
            if t>order+1
                if isempty(no_c), slicer = (t-1):T(in); %slice = (t-order-1):T(in);
                else slicer = (t-1):(no_c(1)+t-2); %slice = (t-order-1):(no_c(1)+t-2);
                end
            else
                if isempty(no_c), slicer = t:T(in); %slice = (t-order):T(in);
                else slicer = t:(no_c(1)+t-2); %slice = (t-order):(no_c(1)+t-2);
                end
            end
            XXt = XX(slicer + s0 - order,:);
            if isnan(C(t,1))
                [gammat,xit,Bt] = nodecluster(XXt,K,hmm,R(slicer,:),in);
                if any(isnan(gammat(:)))
                    error('State time course inference returned NaN - Out of precision?')
                end
            else
                gammat = zeros(length(slicer),K);
                if t==order+1, gammat(1,:) = C(slicer(1),:); end
                xit = zeros(length(slicer)-1, K^2);
                for i=2:length(slicer)
                    gammat(i,:) = C(slicer(i),:);
                    xitr = gammat(i-1,:)' * gammat(i,:) ;
                    xit(i-1,:) = xitr(:)';
                end
                if nargout>=4, Bt = obslike([],hmm,R(slicer,:),XXt,hmm.cache); end
            end
            if t>order+1
                gammat = gammat(2:end,:);
            end
            xi = [xi; xit];
            gamma = [gamma; gammat];
            gammasum = gammasum + sum(gamma);
            if nargout>=4, ll = ll + sum(sum(log(Bt(order+1:end,:)) .* gammat,2)); end
            if nargout>=5, B{in} = [B{in}; Bt(order+1:end,:) ]; end
            if isempty(no_c), break;
            else t = no_c(1)+t-1;
            end
        end
        Gamma{in} = gamma;
        Gammasum(in,:) = gammasum;
        if nargout>=4, LL(in) = ll; end
        %Xi=cat(1,Xi,reshape(xi,T(in)-order-1,K,K));
        Xi{in} = reshape(xi,T(in)-order-1,K,K);
    end
end

% join
Gamma = cell2mat(Gamma);
Xi = cell2mat(Xi);
if n_argout>=5, B  = cell2mat(B); end

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Gamma,Xi,L] = nodecluster(XX,K,hmm,residuals,n)
% inference using normal foward backward propagation

order = hmm.train.maxorder;
T = size(residuals,1) + order;

% if isfield(hmm.train,'grouping') && length(unique(hmm.train.grouping))>1
%     i = hmm.train.grouping(n); 
%     P = hmm.P(:,:,i); Pi = hmm.Pi(:,i)'; 
% else 
%     P = hmm.P; Pi = hmm.Pi;
% end
P = hmm.P; Pi = hmm.Pi;

try
    L = obslike([],hmm,residuals,XX,hmm.cache);
catch
    error('obslike function is giving trouble - Out of precision?')
end

L(L<realmin) = realmin;

if hmm.train.useMEX 
    [Gamma, Xi, scale] = hidden_state_inference_mx(L, Pi, P, order);
    if any(isnan(Gamma(:))) || any(isnan(Xi(:)))
        clear Gamma Xi scale
        warning('hidden_state_inference_mx file produce NaNs - will use Matlab''s code')
    else
        return
    end
end

scale = zeros(T,1);
alpha = zeros(T,K);
beta = zeros(T,K);

alpha(1+order,:) = Pi.*L(1+order,:);
scale(1+order) = sum(alpha(1+order,:));
alpha(1+order,:) = alpha(1+order,:)/scale(1+order);
for i = 2+order:T
    alpha(i,:) = (alpha(i-1,:)*P).*L(i,:);
    scale(i) = sum(alpha(i,:));		% P(X_i | X_1 ... X_{i-1})
    alpha(i,:) = alpha(i,:)/scale(i);
end

scale(scale<realmin) = realmin;

beta(T,:) = ones(1,K)/scale(T);
for i = T-1:-1:1+order
    beta(i,:) = (beta(i+1,:).*L(i+1,:))*(P')/scale(i);
    beta(i,beta(i,:)>realmax) = realmax;
end
Gamma = (alpha.*beta);
Gamma = Gamma(1+order:T,:);
Gamma = rdiv(Gamma,sum(Gamma,2));

Xi = zeros(T-1-order,K*K);
for i = 1+order:T-1
    t = P.*( alpha(i,:)' * (beta(i+1,:).*L(i+1,:)));
    Xi(i-order,:) = t(:)'/sum(t(:));
end

end
