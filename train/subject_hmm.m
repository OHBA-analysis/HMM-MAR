function hmm = subject_hmm(data,T,hmm,Gamma,Xi) %,residuals,XX,XXGXX,Tfactor)
% Get subject-specific states
% If option Xi is specified, it will also update the transition probability
% matrix
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2016)


% set data
N = length(T);
if iscell(data)
    if size(data,1)==1, data = data'; end
    data = cell2mat(data);
end
if iscell(T)
    if size(T,1)==1, T = T'; end
    T = cell2mat(T);
end
if ~isstruct(data), data = struct('X',data); end
if hmm.train.standardise == 1
    for i = 1:N
        t = (1:T(i)) + sum(T(1:i-1));
        data.X(t,:) = data.X(t,:) - repmat(mean(data.X(t,:)),length(t),1);
        data.X(t,:) = data.X(t,:) ./ repmat(std(data.X(t,:)),length(t),1);
    end
end
[hmm.train.orders,order] = formorders(hmm.train.order,hmm.train.orderoffset,...
    hmm.train.timelag,hmm.train.exptimelag);
S = hmm.train.S==1; regressed = sum(S,1)>0;
Sind = formindexes(hmm.train.orders,hmm.train.S); hmm.train.Sind = Sind;
residuals = getresiduals(data.X,T,hmm.train.Sind,hmm.train.maxorder,hmm.train.order,...
    hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag,hmm.train.zeromean);
setxx;

% obtain observation model
hmm = obsupdate (T,Gamma,hmm,residuals,XX,XXGXX,1);

% obtain transition probabilities
if nargin==5
    hmm = hsupdate(Xi,Gamma,T,hmm);
end

end
