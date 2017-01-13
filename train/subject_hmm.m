function hmm = subject_hmm(data,T,hmm,Gamma,Xi)  
% Get subject-specific states
% If argument Xi is provided, it will also update the transition probability
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
if isfield(hmm.train,'A')
    data.X = data.X - repmat(mean(data.X),size(data.X,1),1); % must center
    data.X = data.X * hmm.train.A;
end
if hmm.train.standardise_pc == 1
    data.X = data.X - repmat(mean(data.X),size(data.X,1),1); 
    data.X = data.X ./ repmat(std(data.X),size(data.X,1),1); 
end
hmm.train.orders = formorders(hmm.train.order,hmm.train.orderoffset,...
    hmm.train.timelag,hmm.train.exptimelag);
hmm.train.Sind = formindexes(hmm.train.orders,hmm.train.S); 
if ~hmm.train.zeromean, hmm.train.Sind = [true(1,size(hmm.train.Sind,2)); hmm.train.Sind]; end
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
