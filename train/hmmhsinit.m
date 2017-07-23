function hmm = hmmhsinit(hmm)
% Initialise variables related to the Markov chain
%
% hmm		hmm data structure
%
% OUTPUT
% hmm           hmm structure
%
% Author: Diego Vidaurre, OHBA, University of Oxford

if isfield(hmm.train,'grouping')
    Q = length(unique(hmm.train.grouping));
else
    Q = 1;
end

% Initial state
if Q==1
    hmm.Dir_alpha = ones(1,hmm.K);
    hmm.Pi = ones(1,hmm.K) ./ hmm.K;
else
    hmm.Dir_alpha = ones(hmm.K,Q);
    hmm.Pi = ones(hmm.K,Q) ./ hmm.K;    
end

% State transitions
hmm.Dir2d_alpha = ones(hmm.K,hmm.K,Q);
hmm.P = ones(hmm.K,hmm.K,Q);
for i = 1:Q
    for k = 1:hmm.K
        hmm.Dir2d_alpha(k,k,i) = hmm.train.DirichletDiag;
        hmm.Dir2d_alpha(k,:,i) = hmm.train.PriorWeighting .* hmm.Dir2d_alpha(k,:,i);
        hmm.P(k,:,i) = hmm.Dir2d_alpha(k,:,i) ./ sum(hmm.Dir2d_alpha(k,:,i));
    end
end

% define P-priors
defhmmprior=struct('Dir2d_alpha',[],'Dir_alpha',[]);

defhmmprior.Dir_alpha=ones(1,hmm.K);
defhmmprior.Dir2d_alpha=ones(hmm.K,hmm.K);
for k=1:hmm.K
    defhmmprior.Dir2d_alpha(k,k) = hmm.train.DirichletDiag;
end
defhmmprior.Dir2d_alpha=hmm.train.PriorWeighting.*defhmmprior.Dir2d_alpha;
% assigning default priors for hidden states
if ~isfield(hmm,'prior')
  hmm.prior=defhmmprior;
else
  % priors not specified are set to default
  hmmpriorlist=fieldnames(defhmmprior);
  fldname=fieldnames(hmm.prior);
  misfldname=find(~ismember(hmmpriorlist,fldname));
  for i=1:length(misfldname)
    priorval=getfield(defhmmprior,hmmpriorlist{i});
    hmm.prior=setfield(hmm.prior,hmmpriorlist{i},priorval);
  end
end

end
