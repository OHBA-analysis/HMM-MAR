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

fehist = inf(options.initrep,1);
Gamma = cell(options.initrep,1);

parfor it=1:options.initrep

    opt_worker = options;
    [opt_worker.Gamma,initial_K] = initGamma_random(T-opt_worker.maxorder,opt_worker.K,opt_worker.DirichletDiag);
    hmm0=struct('train',struct());
    hmm0.K = opt_worker.K;
    hmm0.train = options; 
    hmm0.train.Sind = Sind; 
    hmm0.train.cyc = hmm0.train.initcyc;
    hmm0.train.verbose = 0;
    hmm0 = hmmhsinit(hmm0);
    [hmm0,residuals0]=obsinit(data,T,hmm0,opt_worker.Gamma);
    [~,Gamma{it},~,fehist0] = hmmtrain(data,T,hmm0,opt_worker.Gamma,residuals0);
    fehist(it) = fehist0(end);

    if opt_worker.verbose,
        fprintf('Init run %d, %d->%d states, Free Energy = %f \n',it,initial_K,size(Gamma{it},2),fehist(it));
    end

    if size(Gamma{it},2)<opt_worker.K % If states were knocked out, add them back
        Gamma{it} = [Gamma{it} 0.0001*rand(size(Gamma{it},1),opt_worker.K-size(Gamma{it},2))];
        Gamma{it} = Gamma{it} ./ repmat(sum(Gamma{it},2),1,opt_worker.K);
    end

end

[fmin,s] = min(fehist);
Gamma = Gamma{s};

if options.verbose
    fprintf('%i-th was the best iteration with FE=%f \n',s,fmin)
end

end