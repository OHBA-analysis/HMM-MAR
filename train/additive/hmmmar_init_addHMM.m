function Gamma = hmmmar_init_addHMM_simple(data,T,options,Sind)
%
% Initialise the additive hidden Markov chain using iterative vanilla K=2 HMMs
%
% INPUT
% data      observations, a struct with X (time series) and C (classes, optional)
% T         length of observation sequence
% options,  structure with the training options  
% Sind
%
% OUTPUT
% Gamma     p(state given X)
%
% Author: Diego Vidaurre, Aarhus University / Oxford , 2020

if ~isfield(options,'maxorder')
    [~,order] = formorders(options.order,options.orderoffset,options.timelag,options.exptimelag);
    options.maxorder = order;
end
L = options.maxorder; K = options.K; Tall = size(data.X,1)-L*length(T);
if isfield(data,'C'), data = rmfield(data,'C'); end

if options.initTestSmallerK
    warning('Option initTestSmallerK ignored')
end

K = options.K; expanded_K = 2 * K;
options0 = options;
options0.additiveHMM = 0;
options0.verbose = 0;
options0.DirichletDiag = 10;
% get the baseline state
options0.K = 1;
data.C = NaN(Tall,1);
hmm = run_short_hmm(data,T,options0,Sind);
w_bas = hmm.state(1).W.Mu_W(:);

options0.K = expanded_K; 
data.C = NaN(Tall,expanded_K);
cbest = -Inf;

for r = 1:options.initrep
    [hmm,G] = run_short_hmm(data,T,options0,Sind);
    d = zeros(1,expanded_K);
    for k = 1:expanded_K % which state is farthest from baseline?
        if mean(G(:,k)) > 0.001 || sum(G(:,k)) > 100
            w_k = hmm.state(k).W.Mu_W(:);
            d(k) = sum( (w_k - w_bas).^2 );
        end
    end
    [d,ord] = sort(d,'descend');
    c = sum(d(1:K));
    if c > cbest 
        Gamma = G(:,ord(1:K)); 
        cbest = c; 
    end    
    fprintf('Init run %2d, score = %f \n',r,c);
end
    
    
end

function [hmm,Gamma] = run_short_hmm(data,T,options,Sind)

options.additiveHMM = 0; 
keep_trying = true; notries = 0;
while keep_trying
    if options.K > 1
        Gamma = initGamma_random(T-options.maxorder,options.K,...
            min(median(double(T))/10,500));
        if any(~isnan(data.C(:)))
            ind = ~isnan(sum(data.C,2));
            Gamma(ind,:) = data.C(ind,:);
        end
    else
        Gamma = ones(sum(T-options.maxorder),1);
    end
    hmm = struct('train',struct());
    hmm.K = options.K;
    hmm.train = options;
    hmm.train.Sind = Sind;
    hmm.train.cyc = hmm.train.initcyc;
    hmm.train.verbose = 0;
    hmm.train.Pstructure = true(options.K);
    hmm.train.Pistructure = true(1,options.K);
    hmm = hmmhsinit(hmm);
    [hmm,residuals] = obsinit(data,T,hmm,Gamma);
    try
        [hmm,Gamma,~,fehist] = hmmtrain(data,T,hmm,Gamma,residuals);
        keep_trying = false;
    catch
        notries = notries + 1; 
        if notries > 10, error('Initialisation went wrong'); end
        disp('Something strange happened in the initialisation - repeating')
    end
end

end
