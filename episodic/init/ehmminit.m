function Gamma = ehmminit(data,T,options)
%
% Initialise the ehmm chain using iterative vanilla K=2 HMMs
%
% INPUT
% data      observations, a struct with X (time series) and C (classes, optional)
% T         length of observation sequence
% options   structure with the training options
% Sind
%
% OUTPUT
% Gamma     p(state given X)
%
% Author: Diego Vidaurre, Aarhus University / Oxford , 2020

if ~isfield(options,'maxorder')
    [~,order] = formorders(options.order,options.orderoffset,...
        options.timelag,options.exptimelag);
    options.maxorder = order;
end
K = options.K;
if isfield(options,'initTestSmallerK') && options.initTestSmallerK
    warning('Option initTestSmallerK ignored')
end
% provided_baseline = ...
%     (isfield(options,'ehmm_baseline_data') && ~isempty(options.ehmm_baseline_data)) ...
%     || (isfield(options,'ehmm_baseline_w') && ~isempty(options.ehmm_baseline_w));
% r = 1; stop = false;

if isfield(options,'ehmm_baseline_data') && ~isempty(options.ehmm_baseline_data)
    baseline = computeBaseline(options);
elseif (isfield(options,'ehmm_baseline_w') && ~isempty(options.ehmm_baseline_w))
    if isstruct(options.ehmm_baseline_w)
        baseline = options.ehmm_baseline_w;
    else
        baseline = struct(); baseline.Mu_W = options.ehmm_baseline_w;
    end
else
    baseline = computeBaseline(options,data.X,T);
end

[hmm,I,G] = run_short_hmm_batches(data,T,options,K);
Gamma = findStandoutChains(T-options.order,hmm,baseline,I,G,K);

end


function [hmm,I,G] = run_short_hmm_batches(data,T,options,K)
L = 10000; step = 2000; threshold = 20; 
N = length(T); order = options.order;
hmm = []; I = []; G = []; 
for j = 1:N
    t0 = sum(T(1:j-1)); t0g = t0 - (j-1) * order; 
    t =  t0 + (1:(L+order)); tg = t0g + (1:L);
    while t(end) <= T
        dat = data; dat.X = dat.X(t,:);
        [hm,g] = run_short_hmm(dat,length(t),options,K);
        if isempty(hmm)
            hmm = hm;
        else
            for k = 1:length(hm.state)
                if sum(g(:,k))>threshold
                    hmm.state(end+1) = hm.state(k);
                end
            end
        end
        for k = 1:length(hm.state)
            if sum(g(:,k))>threshold
                I = [I; [tg(1) tg(end)]];
                G = [G single(g(:,k))];
            end
        end
        t = t + step; tg = tg + step;
    end
end
end


function [hmm,Gamma] = run_short_hmm(data,T,options,K)
options.K = K;
GammaInit = initGamma_random(T-options.maxorder,K,...
    min(median(double(T))/10,500));
hmm = struct('train',struct());
hmm.K = K;
hmm.train = options;
hmm.train.ndim = size(data.X,2);
hmm.train.cyc = hmm.train.cyc;
hmm.train.verbose = 0; %%%%
hmm.train.episodic = 0;
hmm.train.Pstructure = true(options.K);
hmm.train.Pistructure = true(1,options.K);
hmm.train.stopcriterion = 'FreeEnergy'; 
if isfield(options,'DirichletDiag_Init')
    hmm.train.DirichletDiag = options.DirichletDiag_Init;
end
hmm = hmmhsinit(hmm);
if isfield(options,'DirichletDiag_Init')
    hmm.train.DirichletDiag = options.DirichletDiag;
end
[hmm,residuals] = obsinit(data,T,hmm,GammaInit);
data.C = NaN(size(data.C,1),hmm.K);
[hmm,Gamma] = hmmtrain(data,T,hmm,GammaInit,residuals);
end


function Gamma = findStandoutChains(T,hmm,baseline,I,G,nchains)
K = length(hmm.state);
hmm.state(K+1).W = baseline;
hmm.train.Pstructure = true(K+1); hmm.train.Pistructure = true(1,K+1);
hmm.train.stopcriterion = 'FreeEnergy'; 
fit = hmmspectramar([],[],hmm);
c = zeros(K,1);
ndim = size(fit.state(K+1).psd,2);
psd = []; for n = 1:ndim, psd = [psd; fit.state(K+1).psd(:,n,n)]; end
psd = psd ./ sum(psd);
W = zeros(length(baseline.Mu_W),K);
PSD = zeros(length(psd),K);
for k = 1:K
    psdk = []; for n = 1:ndim, psdk = [psdk; fit.state(k).psd(:,n,n)]; end
    psdk = psdk ./ sum(psdk);
    c(k) = corr(psdk,psd); 
    W(:,k) = hmm.state(k).W.Mu_W;
    PSD(:,k) = psdk;
end
[~,jj] = min(c);
while length(jj) < nchains
   for k = 1:K
      c(k) = max(c(k),corr(PSD(:,k),PSD(:,jj(end)))); 
   end
   [~,j] = min(c); jj = [jj j]; 
end
Gamma = zeros(sum(T),nchains);
for k = 1:nchains
   Gamma(I(jj(k),1):I(jj(k),2),k) = G(:,jj(k)); 
end
end

% 
% function Gamma = estimateGamma(data,T,W,Wbaseline,options)
% orders = formorders(options.order,options.orderoffset,options.timelag,options.exptimelag);
% [XX,Y] = formautoregr(data.X,T,orders,orders(end),1);
% K = size(W,2);
% % r = zeros(size(Y,1),K+1);
% % for k = 1:K
% %     r(:,k) = (XX * W(:,k) - Y).^2;
% % end
% % r(:,K+1) = (XX * Wbaseline - Y).^2;
% ehmm = struct('train',struct());
% ehmm.K = options.K;
% ehmm.train = options;
% ehmm = hmmhsinit(ehmm,[],T);
% for k = 1:K
%     ehmm.state(k).W.Mu_W = W(:,k);
% end
% ehmm.state(K+1).W.Mu_W = Wbaseline;
% ehmm.Omega = struct('Gam_rate',1,'Gam_shape',1);
% np = size(W,1); ndim = 1;
% for n = 1:ndim
%     ehmm.state_shared(n).Mu_W = zeros((K+1)*np,1);
% end
% for k = 1:K+1
%     ind = (k-1)*np + (1:np);
%     for n = 1:ndim
%         ehmm.state_shared(n).Mu_W(ind) = ehmm.state(k).W.Mu_W;
%     end
% end
% Gamma = hsinference(data,T,ehmm,Y,options,XX,zeros(sum(T)-length(T)*options.order,K));
% end
% 
% 

