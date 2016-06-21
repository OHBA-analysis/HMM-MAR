function Gamma = hmmmar_init(data,T,options,Sind)
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

if isfield(options,'maxorder')
    order = options.maxorder;
else 
    order = options.order;
end

% Run two initializations for each K less than requested K, plus options.initrep K
init_k = [repmat(1:(options.K-1),1,2) options.K*ones(1,options.initrep)];
init_k = init_k(end:-1:1);
p = options.DirichletDiag/(options.DirichletDiag + options.K - 1); % Probability of remaining in same state
step_lifetime =  (1 + -(p-1)*p/((log(p)^2))); % Expected number of steps for this Dirichletdiag and k

fehist = inf(length(init_k),1);
Gamma = cell(length(init_k),1);

parfor it=1:length(init_k)

    opt_worker = options;
    opt_worker.K = init_k(it);
    opt_worker.DirichletDiag = compute_dirichletdiag(step_lifetime/options.Fs,options.Fs,opt_worker.K);
    % fprintf('Changed DD from %d to %d for K=%d\n',options.DirichletDiag,opt_worker.DirichletDiag,opt_worker.K)


    data2 = data;
    data2.C = data2.C(:,1:opt_worker.K);

    opt_worker.Gamma = initGamma_random(T-opt_worker.maxorder,opt_worker.K,1);

    hmm0=struct('train',struct());
    hmm0.K = opt_worker.K;
    hmm0.train = options; 
    hmm0.train.Sind = Sind; 
    hmm0.train.cyc = hmm0.train.initcyc;
    hmm0.train.verbose = 0;
    hmm0 = hmmhsinit(hmm0);
    [hmm0,residuals0]=obsinit(data2,T,hmm0,opt_worker.Gamma);
    [~,Gamma{it},~,fehist0] = hmmtrain(data2,T,hmm0,opt_worker.Gamma,residuals0);
    fehist(it) = fehist0(end);

    if opt_worker.verbose,
        fprintf('Init run %d, %d->%d states, Free Energy = %f \n',it,opt_worker.K,size(Gamma{it},2),fehist(it));
    end

    if size(Gamma{it},2)<options.K % If states were knocked out, add them back
        Gamma{it} = [Gamma{it} 0.0001*rand(size(Gamma{it},1),options.K-size(Gamma{it},2))];
        Gamma{it} = bsxfun(@rdivide,Gamma{it},sum(Gamma{it},2));
    end

end

[fmin,s] = min(fehist);
Gamma = Gamma{s};

if options.verbose
    fprintf('%i-th was the best iteration with FE=%f \n',s,fmin)
end

end