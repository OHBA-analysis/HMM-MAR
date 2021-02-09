function Gamma = hmmmar_init_ness(data,T,options,Sind)
%
% Initialise the NESS chain using iterative vanilla K=2 HMMs
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
    [~,order] = formorders(options.order,options.orderoffset,...
        options.timelag,options.exptimelag);
    options.maxorder = order;
end
L = options.maxorder; K = options.K; 
Tall = size(data.X,1)-L*length(T);
if isfield(data,'C'), data = rmfield(data,'C'); end

% Run two initializations for each K less than requested K, plus options.initrep K
if options.initTestSmallerK
    warning('Option initTestSmallerK ignored')
end

cbest = -Inf;

r = 1; badinits = 0; maxbadinits = options.initrep * 5; 

while r <= options.initrep
        
    options0 = options;
    options0.K = 1;
    options0.nessmodel = 0;
    options0.verbose = 0;
    options0.Pstructure = true;
    options0.Pistructure = true;
    options0.DirichletDiag = 10; 
    % get the baseline state
    data.C = NaN(Tall,1);
    hmm = run_short_hmm(data,T,options0,Sind);
    w_bas = hmm.state(1).W.Mu_W(:);
    data.C = NaN(Tall,2);
    for k = 1:K
        options0.K = k + 1;
        options0.Pistructure = true(1,k+1);
        options0.Pstructure = true(k+1); 
        if k > 1 
            options0.Pstructure(k:k+1,1:k-1) = false; 
            options0.Pstructure(1:k-1,1:k-1) = eye(k-1);
        end
        % run with 2 free-moving states
        [hmm,G] = run_short_hmm(data,T,options0,Sind);
        d = zeros(1,k+1);
        for l = k:k+1 % which state is farthest from baseline?
            w_k = hmm.state(l).W.Mu_W(:);
            d(l) = sum( (w_k - w_bas).^2 );
        end
        [~,kk] = max(d);
        % fix that state for the next states
        data.C = [data.C zeros(Tall,1)];
        ind = isnan(sum(data.C,2));
        data.C(ind,end) = NaN;
        ind = G(:,kk) > 0.75;
        data.C(ind,:) = 0; data.C(ind,k) = 1;
    end
    Gamma = data.C(:,1:K);
    Gamma(isnan(Gamma)) = 0;
    
    if any(sum(Gamma)==0)
        disp('Bad initialisation')
        badinits = badinits + 1;
        if badinits > maxbadinits 
            error(['Was not able to initialize ' num2str(K) ...
                ' chains. Rerun or decrease K'])
        end
        continue
    end
    
    c = 0;
    for k = 1:K % which state is farthest from baseline?
        w_k = hmm.state(k).W.Mu_W(:);
        c = c + sum( (w_k - w_bas).^2 );
    end
    if c > cbest, Gammabest = Gamma; cbest = c; end    
    fprintf('Init run %2d, score = %f \n',r,c);
    r = r + 1;
    %figure(10+r);area(Gamma);title(num2str(c))
    
end

Gamma = Gammabest;
if any(sum(Gamma)==0)
    error(['Was not able to initialize ' num2str(K) ' chains. Rerun or decrease K'])
end

end

function [hmm,Gamma,fehist] = run_short_hmm(data,T,options,Sind)

options.nessmodel = 0; 
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
    hmm.train.verbose = 0; %%%%
    %hmm.train.Pstructure = true(options.K);
    %hmm.train.Pistructure = true(1,options.K);
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
