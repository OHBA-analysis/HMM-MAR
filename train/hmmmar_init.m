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


useParallel = options.useParallel;
options.useParallel = 0;

if useParallel
    
    Fehist = zeros(options.initrep,1);
    GammaList = cell(options.initrep,1);
    parfor it=1:options.initrep
        opt = options;
        opt.Gamma = initGamma_random(T-opt.maxorder,opt.K,opt.DirichletDiag);
        hmm0=struct('train',struct());
        hmm0.K = opt.K;
        hmm0.train = opt;
        hmm0.train.Sind = Sind;
        hmm0.train.cyc = hmm0.train.initcyc;
        hmm0.train.verbose = 0;
        hmm0 = hmmhsinit(hmm0);
        [hmm0,residuals0] = obsinit(data,T,hmm0,opt.Gamma);
        [~,Gamma0,~,fehist] = hmmtrain(data,T,hmm0,opt.Gamma,residuals0);
        Fehist(it) = fehist(end);
        if size(Gamma0,2)<opt.K
            Gamma0 = [Gamma0 0.0001*rand(size(Gamma0,1),opt.K-size(Gamma0,2))];
            Gamma0 = Gamma0 ./ repmat(sum(Gamma0,2),1,opt.K);
        end
        GammaList{it} = Gamma0;
        if opt.verbose,
            fprintf('Init run %d, Free Energy %f \n',it,Fehist(it));
        end
    end
    [fehist,it] = min(Fehist);
    Gamma = GammaList{it};
    if options.verbose
        fprintf('%i-th was the best iteration with FE=%f \n',it,fehist)
    end
    
else
    
    fehist = Inf;

    for it=1:options.initrep
        options.Gamma = initGamma_random(T-options.maxorder,options.K,options.DirichletDiag);
        hmm0=struct('train',struct());
        hmm0.K = options.K;
        hmm0.train = options;
        hmm0.train.Sind = Sind;
        hmm0.train.cyc = hmm0.train.initcyc;
        hmm0.train.verbose = 0;
        hmm0 = hmmhsinit(hmm0);
        [hmm0,residuals0] = obsinit(data,T,hmm0,options.Gamma);
        [~,Gamma0,~,fehist0] = hmmtrain(data,T,hmm0,options.Gamma,residuals0);
        if size(Gamma0,2)<options.K
            Gamma0 = [Gamma0 0.0001*rand(size(Gamma0,1),options.K-size(Gamma0,2))];
            Gamma0 = Gamma0 ./ repmat(sum(Gamma0,2),1,options.K);
        end
        if options.verbose,
            fprintf('Init run %d, Free Energy %f \n',it,fehist0(end));
        end
        if fehist0(end)<fehist(end),
            fehist = fehist0; Gamma = Gamma0; s = it;
        end
    end
    if options.verbose
        fprintf('%i-th was the best iteration with FE=%f \n',s,fehist(end))
    end
    
end


end