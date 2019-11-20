function [Gamma,LL,B] = clustinference(data,T,hmm,residuals,options,XX)
%
% inference of the latent variables for the model with no state switching
%
% INPUT
%
% data      Observations - a struct with X (time series) and C (classes)
% T         Number of time points for each time series
% hmm       hmm data structure (which is not an actual hmm in this case)
% residuals in case we train on residuals, the value of those.
% XX        optionally, XX, as computed by setxx.m, can be supplied
%
% OUTPUT
%
% Gamma     Probability of hidden state given the data
% Gammasum  sum of Gamma over t
% LL        Log-likelihood
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2019)

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

if ~isfield(hmm,'Pi')
    hmm = hmmhsinit(hmm);
end

if nargin<6 || isempty(XX)
    setxx;
end

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
    elseif strcmp(train.covtype,'diag')
        ldetWishB=0;
        PsiWish_alphasum = 0;
        for n=1:ndim
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
    end
    hmm.cache.ldetWishB{k} = ldetWishB;
    hmm.cache.PsiWish_alphasum{k} = PsiWish_alphasum;
    hmm.cache.C{k} = C;
end

LL = zeros(N,1);
Pi = zeros(N,K);

if hmm.train.useParallel==1 && N>1
     parfor j = 1:N
         t0 = sum(T(1:j-1)); s0 = t0 - order*(j-1);
         t = order+1;
         slicer = (t-1):T(j);
         XXt = XX(slicer + s0 - order,:); 
         L = obslike([],hmm,R(slicer,:),XXt,hmm.cache);
         S1 = sum(log(L)); 
         S2 = 
         
         Pi(j,
         
         LL(j,:) = sum(log(L))
     end
    
else
    
end


Gamma = zeros(N,K);




