function [hmm,Gamma,fehist] = hmmmar_init(data,T,options,Sind)
%
% Initialise the hidden Markov chain using HMM-MAR
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
% Author: Diego Vidaurre, University of Oxford
% Author: Romesh Abeysuriya, University of Oxford

if ~isfield(options,'maxorder')
    [~,order] = formorders(options.order,options.orderoffset,options.timelag,options.exptimelag);
    options.maxorder = order; 
end

if options.initTestSmallerK % Run two initializations for each K less than requested K, plus options.initrep K
    init_k = [repmat(1:(options.K-1),1,2) options.K*ones(1,options.initrep)];
    init_k = init_k(end:-1:1);
else % Standard behaviour, test specified K options.initrep times
    init_k = options.K*ones(1,options.initrep);
end

felast = zeros(length(init_k),1);
maxfo = zeros(length(init_k),1);
fehist = cell(length(init_k),1);
Gamma = cell(length(init_k),1);
hmm = cell(length(init_k),1);

if options.useParallel && length(init_k) > 1 % not very elegant
    parfor it = 1:length(init_k)
        [hmm{it},Gamma{it},fehist{it}] = run_initialization(data,T,options,Sind,init_k(it));
        felast(it) = fehist{it}(end);
        maxfo(it) = mean(getMaxFractionalOccupancy(Gamma{it},T,options));
        if options.verbose
            if options.nessmodel
                fprintf('Init run %2d, Free Energy = %f \n',it,felast(it));
            else
                fprintf('Init run %2d, %2d->%2d states, Free Energy = %f \n',...
                    it,init_k(it),size(Gamma{it},2),felast(it));
            end
        end
    end
else
    for it = 1:length(init_k)
        [hmm{it},Gamma{it},fehist{it}] = run_initialization(data,T,options,Sind,init_k(it));
        felast(it) = fehist{it}(end);
        maxfo(it) = mean(getMaxFractionalOccupancy(Gamma{it},T,options));
        if options.verbose
            if options.nessmodel
                fprintf('Init run %2d, Free Energy = %f \n',it,felast(it));
            else
                fprintf('Init run %2d, %2d->%2d states, Free Energy = %f \n',...
                    it,init_k(it),size(Gamma{it},2),felast(it));
            end
        end
    end 
end

if isfield(options,'initcriterion') && strcmpi(options.initcriterion,'FreeEnergy')
    [fe,s] = min(felast);
    if options.verbose
        fprintf('%i-th was the best iteration with FE=%f \n',s,fe)
    end
else
    [fo,s] = min(maxfo);
    if options.verbose
        fprintf('%i-th was the best iteration with mean maxFO=%f \n',s,fo)
    end    
end

Gamma = Gamma{s};
hmm = hmm{s};
fehist = fehist{s};

end

function [hmm,Gamma,fehist] = run_initialization(data,T,options,Sind,init_k)
% INPUTS
% - data,T,options,Sind <same as hmmmar_init>
% - init_k is the number of states to use for this initialization

% Need to adjust the worker dirichletdiags if testing smaller K values
%if ~options.nessmodel && init_k < options.K
if init_k < options.K
    for j = 1:length(options.DirichletDiag)
        p = options.DirichletDiag(j)/(options.DirichletDiag(j) + options.K - 1); % Probability of remaining in same state
        f_prob = dirichletdiags.mean_lifetime(); % Function that returns the lifetime in steps given the probability
        expected_lifetime =  f_prob(p)/options.Fs; % Expected number of steps given the probability
        options.K = init_k;
        adjusted_DirichletDiag = dirichletdiags.get(expected_lifetime,options.Fs,options.K);
        if isfinite(adjusted_DirichletDiag) % It is NaN if there was a numerical issue
            options.DirichletDiag(j) = adjusted_DirichletDiag;
        end
    end
end

data.C = data.C(:,1:options.K);
% Note - initGamma_random uses DD=1 so that there are lots of transition times, which
% helps the inference not get stuck in a local minimum. options.DirichletDiag is
% then used inside hmmtrain when computing the free energy
keep_trying = true; notries = 0; 
while keep_trying
    Gamma = initGamma_random(T-options.maxorder,options.K,...
        min(median(double(T))/10,500),...
        options.Pstructure,options.Pistructure,...
        options.nessmodel);
    hmm = struct('train',struct());
    hmm.K = options.K;
    hmm.train = options;
    hmm.train.Sind = Sind;
    hmm.train.cyc = max(hmm.train.initcyc,2);
    hmm.train.verbose = 0;
    hmm.train.plotGamma = 0;
    hmm = hmmhsinit(hmm);
    if isfield(hmm.train,'Gamma'), hmm.train = rmfield(hmm.train,'Gamma'); end
    [hmm,residuals] = obsinit(data,T,hmm,Gamma);
    try
        [hmm,Gamma,~,fehist] = hmmtrain(data,T,hmm,Gamma,residuals);
        fehist(end) = [];
        keep_trying = false;
    catch
        notries = notries + 1; 
        if notries > 10, error('Initialisation went wrong'); end
        disp('Something strange happened in the initialisation - repeating')
    end
    hmm.train.verbose = options.verbose;
    hmm.train.plotGamma = options.plotGamma;
end
%fe = fehist(end);
end
