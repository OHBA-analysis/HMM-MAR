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

if isfield(options,'ehmm_init_from_hmm')
    hmm = options.ehmm_init_from_hmm.hmm;
    Gamma = options.ehmm_init_from_hmm.Gamma;
    if size(Gamma,2) ~= (K+1)
        error('Init Gamma must have K+1 states')
    end
    if ~strcmp(hmm.train.covtype,'uniquediag')
        error('Covtype of init HMM must be uniquediag')
    end
else
    [hmm,Gamma] = run_short_hmm(data,T,options,K+1);
end

k = findBaseline(hmm,baseline);
disp(['Baseline state: ' num2str(k)])
Gamma = Gamma(:,setdiff(1:K+1,k));

% while ~stop
%     if provided_baseline
%         [ehmm,Gamma] = run_short_hmm(data,T,options,K+1);
%         I = findBaseline(ehmm,baseline);
%         Gamma = Gamma(:,setdiff(1:K+1,I));
%         stop = true;
%     else  
%         [ehmm,Gammar] = run_short_hmm(data,T,options,K*2);
%         I = selectStates(data,T,Gammar,ehmm,K);
%         Gammar = Gammar(:,I);
%         if length(I) < K
%             disp(['Rep : ' num2str(r) '. Unable to initialize ' num2str(K) ...
%                 ' chains; only got ' num2str(length(I)) ])
%             if size(Gammar,2) > size(Gamma,2), Gamma = Gammar; end
%             r = r + 1;
%             if r == 10, error('Must stop here'); end
%         else
%             Gamma = Gammar;
%             stop = true;
%         end
%     end
% end

end


function [hmm,Gamma] = run_short_hmm(data,T,options,K)
options.K = K;
if options.order > 0 && options.zeromean
    GammaInit = initGamma_window(data,T,options);
else
    GammaInit = initGamma_random(T-options.maxorder,K,...
        min(median(double(T))/10,500));
end
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


function I = findBaseline(hmm,baseline)
K = length(hmm.state);
hmm.state(K+1).W = baseline;
hmm.train.Pstructure = true(K+1); hmm.train.Pistructure = true(1,K+1);
hmm.train.stopcriterion = 'FreeEnergy'; 
fit = hmmspectramar([],[],hmm);
d = zeros(K,1);
ndim = size(fit.state(K+1).psd,2);
psd = []; for n = 1:ndim, psd = [psd; fit.state(K+1).psd(:,n,n)]; end
psd = psd ./ sum(psd);
for k = 1:K
    psdk = []; for n = 1:ndim, psdk = [psdk; fit.state(k).psd(:,n,n)]; end
    psdk = psdk ./ sum(psdk);
    d(k) = sum( (psdk - psd).^2);
end
[~,I] = min(d);
end



% function I = selectStates(data,T,G,hmm,K)
% lambda = hmm.train.ehmm_regularisation_baseline;
% Kmax = size(G,2);
% if Kmax <= K, I = 1:Kmax; return; end
% if isfield(hmm.train,'ehmm_baseline_data')
%     XXb = formautoregr(hmm.train.ehmm_baseline_data.X,hmm.train.ehmm_baseline_data.T,...
%         hmm.train.orders,hmm.train.maxorder,hmm.train.zeromean);
%     residualsb =  getresiduals(hmm.train.ehmm_baseline_data.X,hmm.train.ehmm_baseline_data.T,...
%         hmm.train.Sind,hmm.train.maxorder,hmm.train.order,...
%         hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag,hmm.train.zeromean);
%     gram = (XXb' * XXb);
%     gram = (gram + gram') / 2 ;
%     gram = gram + trace(gram) * lambda * eye(size(gram,2));
%     igram = inv(gram);
%     Mu_W = igram * (XXb' * residualsb);
%     provided_baseline = true;
% elseif isfield(hmm.train,'ehmm_baseline_w')
%     Mu_W = hmm.train.ehmm_baseline_w.Mu_W;
%     provided_baseline = true;
% else
%     provided_baseline = false;
% end
% %%% choose baseline
% use = true(Kmax,1);
% if provided_baseline
%     d = zeros(Kmax,1);
%     for k = 1:Kmax
%         d(k) = sum((hmm.state(k).W.Mu_W(:) - Mu_W(:)).^2);
%     end
%     [~,k] = min(d);
%     use(k) = false;
% end
% %%% Selecting the states that are not well explained by others
% % put the data in the right format
% orders = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
% residuals = getresiduals(data.X,T,hmm.train.Sind,hmm.train.maxorder,hmm.train.order,...
%     hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag,hmm.train.zeromean);
% XX = formautoregr(data.X,T,orders,hmm.train.maxorder,hmm.train.zeromean);
% p = size(residuals,2);
% % regression coefficients
% W = zeros([size(hmm.state(1).W.Mu_W) Kmax]);
% for k = 1:Kmax, W(:,:,k) = hmm.state(k).W.Mu_W; end
% r2 = zeros(Kmax,1);
% for k = 1:Kmax
%     if ~use(k), continue; end
%     ind = G(:,k)>0.75; Nk = sum(ind);
%     if Nk < 10, continue; end
%     y = residuals(ind,:); y = y(:);
%     % prediction within state parameters
%     yhat = XX(ind,:) * hmm.state(k).W.Mu_W; yhat = yhat(:);
%     r2(k) = 1 - sum((yhat-y).^2) / sum(y.^2);
%     % prediction by others
%     yhat = ones(Nk,p,Kmax);
%     kk = setdiff(1:Kmax,k);
%     for ik2 = 1:Kmax-1  % last column of ones
%         yhat(:,:,ik2) = XX(ind,:) * hmm.state(kk(ik2)).W.Mu_W;
%     end
%     yhat = reshape(yhat,Nk*p,Kmax); % includes intercept
%     opts1 = optimset('display','off');
%     w = lsqlin(yhat,y,[],[],ones(1,Kmax),1,zeros(Kmax,1),ones(Kmax,1),[],opts1);
%     yhat = yhat * w;
%     r2k = 1 - sum((yhat-y).^2) / sum((y-mean(y)).^2);
%     r2(k) = r2(k) * (r2(k) - max(r2k,0));
% end
% fo = sum(G);
% r2(fo<100) = 0; % ignore those with too few points
% [~,I] = sort(r2,'descend');
% I = I(1:K);
% end

% function I = selectStates(G,hmm,K)
% % Selecting the states that are not linear combination of the others,
% %  and with not too low FO
% Khmm = size(G,2);
% if Khmm <= K, I = 1:Khmm; return; end
% bn = dec2bin(1:2^(Khmm-1)); bn = bn(1:end-1,2:end)';
% D = zeros(size(bn)); % K x ncomb
% for ik = 1:size(bn,2)
%     D(:,ik) = str2num(bn(:,ik));
% end
% B = zeros(length(hmm.state(1).W.Mu_W(:)),Khmm); % p x K
% for k = 1:Khmm, B(:,k) = hmm.state(k).W.Mu_W(:); end
% err = zeros(Khmm,size(bn,2)); % states by combination of states
% for ik = 1:Khmm
%     Bhat = B(:,setdiff(1:Khmm,ik)) * D;
%     err(ik,:) = mean( abs(Bhat - repmat(B(:,ik),1,size(D,2))) );
% end
% err = min(err,[],2); % how well is predicted by a sum of others
% fo = sum(G); %l1 = mean(abs(B));
% kept = Khmm; % no. of kept states
% % % remove the one with the lowest betas
% % [v,ik] = min(l1);
% % if v < 0.1 * median(l1(setdiff(1:Khmm,ik)))
% %     err(ik) = -Inf; fo(ik) = Inf;
% %     kept = kept-1;
% % end
% % remove the ones with too low FO
% while true
%     if kept < K, break; end
%     if ~any(fo<1e-3), break; end
%     [v,ik] = min(fo);
%     if v < 50, err(ik) = -Inf; fo(ik) = Inf; kept = kept - 1; end
% end
% % choose the ones with the largest error (less well predicted by others)
% [~,I] = sort(err,'descend');
% I = I(1:K);
% end
