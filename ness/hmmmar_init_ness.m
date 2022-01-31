function Gamma = hmmmar_init_ness(data,T,options)
%
% Initialise the NESS chain using iterative vanilla K=2 HMMs
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

Gamma = []; 
r = 1; 

while true
    [hmm,Gammar] = run_short_hmm(data,T,options);
    I = selectStates(Gammar,hmm,K);
    Gammar = Gammar(:,I);
    if length(I) < K
        disp(['Rep : ' num2str(r) '. Unable to initialize ' num2str(K) ...
            ' chains; only got ' num2str(length(I)) ])
        if size(Gammar,2) > size(Gamma,2), Gamma = Gammar; end
        r = r + 1; 
    else
        Gamma = Gammar;
        break
    end
end
 
end

function [hmm,Gamma] = run_short_hmm(data,T,options)
options.K = 2 * options.K;
if options.order > 0 && options.zeromean
    Gamma = initGamma_window(data,T,options);
else
    Gamma = initGamma_random(T-options.maxorder,options.K*2,...
        min(median(double(T))/10,500));
end
hmm = struct('train',struct());
hmm.K = options.K;
hmm.train = options;
hmm.train.ndim = size(data.X,2);
hmm.train.cyc = hmm.train.cyc;
hmm.train.verbose = 0; %%%%
hmm.train.nessmodel = 0;
hmm.train.Pstructure = true(options.K);
hmm.train.Pistructure = true(1,options.K);
hmm = hmmhsinit(hmm);
[hmm,residuals] = obsinit(data,T,hmm,Gamma);
data.C = NaN(size(data.C,1),hmm.K);
[hmm,Gamma] = hmmtrain(data,T,hmm,Gamma,residuals);
end


function I = selectStates(G,hmm,K)
% Selecting the states that are not linear combination of the others,
%  and with not too low FO
Khmm = size(G,2); 
if Khmm <= K, I = 1:Khmm; return; end
bn = dec2bin(1:2^(Khmm-1)); bn = bn(1:end-1,2:end)';
D = zeros(size(bn)); % K x ncomb
for ik = 1:size(bn,2)
    D(:,ik) = str2num(bn(:,ik));
end
B = zeros(length(hmm.state(1).W.Mu_W(:)),Khmm); % p x K
for k = 1:Khmm, B(:,k) = hmm.state(k).W.Mu_W(:); end
err = zeros(Khmm,size(bn,2));
for ik = 1:Khmm
    Bhat = B(:,setdiff(1:Khmm,ik)) * D;
    err(ik,:) = mean( abs(Bhat - repmat(B(:,ik),1,size(D,2))) );
end
err = min(err,[],2); % how well is predicted by a sum of others  
fo = mean(G); l1 = mean(abs(B));
kept = Khmm; % no. of kept states
% remove the one with the lowest betas
[v,ik] = min(l1);
if v < 0.1 * median(l1(setdiff(1:Khmm,ik)))
    err(ik) = -Inf; fo(ik) = Inf; 
    kept = kept-1; 
end
% remove the ones with too low FO
while true
    if kept < K, break; end
    if ~any(fo<1e-3), break; end
    [v,ik] = min(fo); 
    if v < 1e-3, err(ik) = -Inf; fo(ik) = Inf; kept = kept - 1; end
end
% choose the ones with the largest error (less well predicted by others)    
[~,I] = sort(err,'descend'); 
I = I(1:K);
end


% figure(1)
% subplot(221); bar(err);
% subplot(222); bar(l1);
% subplot(2,2,[3 4])
% imagesc(G(1:500,:)')