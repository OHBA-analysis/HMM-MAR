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

if isfield(options,'maxorder'), order = options.maxorder;
else order = options.order;
end

fehist = Inf;
for it=1:options.initrep
    options.Gamma = initGamma_random(T-options.maxorder,options.K,options.DirichletDiag);
    hmm0=struct('train',struct());
    hmm0.K = options.K;
    hmm0.train = options; 
    hmm0.train.Sind = Sind; 
    hmm0.train.cyc = hmm0.train.initcyc;
    hmm0.train.verbose = 0;
    hmm0=hmmhsinit(hmm0);
    [hmm0,residuals0]=obsinit(data,T,hmm0,options.Gamma);
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