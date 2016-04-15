function Path = hmmdecode(X,T,hmm,type,residuals,options,markovTrans)
%
% State time course and Viterbi decoding for hmm
% The algorithm is run for the whole data set, including those whose class
% was fixed. This means that the assignment for those can be different.
%
% INPUT
% X             Observations
% T             length of series
% hmm           hmm data structure
% residuals     in case we train on residuals, the value of those (optional)
% options       the hmm options, that will be used if hmm.train is missing
% type: 0, state time courses; 1, viterbi path
%
% OUTPUT
% vpath         (T x 1) maximum likelihood state sequence (type=0 OR
% vpath         (T x K) state time courses
%
% Author: Diego Vidaurre, OHBA, University of Oxford

if nargin<4, type = 0; end
if nargin<5, residuals = []; end
if nargin<6, options = []; end
if nargin<7, markovTrans = []; end

if isfield(hmm.train,'BIGNbatch') && hmm.train.BIGNbatch < length(T);
    Path = hmmsdecode(X,T,hmm,type,markovTrans);
    return
end

if iscell(T)
    for i = 1:length(T)
        if size(T{i},1)==1, T{i} = T{i}'; end
    end
    if size(T,1)==1, T = T'; end
    T = cell2mat(T);
end
if iscell(X)
    X = cell2mat(X);
end

if type==0
   Path = hsinference(X,T,hmm,residuals,options); 
   return
end

if ~isfield(hmm,'train')
    if nargin<5, error('You must specify the field options if hmm.train is missing'); end
    hmm.train = checkoptions(options,X,T,0);
end

N = length(T);
K = length(hmm.state);

if isempty(residuals)
    if ~isfield(hmm.train,'Sind')
        orders = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
        hmm.train.Sind = formindexes(orders,hmm.train.S);
    end
    residuals =  getresiduals(X,T,hmm.train.Sind,hmm.train.maxorder,hmm.train.order,...
        hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag,hmm.train.zeromean);
end

if ~isfield(hmm,'P')
    hmm = hmmhsinit (hmm);
end
    
P = hmm.P;
Pi = hmm.Pi;

if ~hmm.train.multipleConf
    [~,order] = formorders(hmm.train.order,hmm.train.orderoffset,...
        hmm.train.timelag,hmm.train.exptimelag);
else
    order = hmm.train.maxorder;
end

Path = zeros(sum(T)-length(T)*order,1);
tacc = 0;

for tr=1:N
    
    q_star = ones(T(tr)-order,1);
    
    alpha=zeros(T(tr),K);
    beta=zeros(T(tr),K);
    
    % Initialise Viterbi bits
    delta=zeros(T(tr),K);
    psi=zeros(T(tr),K);
    
    if tr==1, t0 = 0; s0 = 0;
    else t0 = sum(T(1:tr-1)); s0 = t0 - order*(tr-1);
    end
    
    B = obslike(X(t0+1:t0+T(tr),:),hmm,residuals(s0+1:s0+T(tr)-order,:));
    B(B<realmin) = realmin;
    
    scale=zeros(T(tr),1);
    % Scaling for delta
    dscale=zeros(T(tr),1);
    
    alpha(1+order,:)=Pi(:)'.*B(1+order,:);
    scale(1+order)=sum(alpha(1+order,:));
    alpha(1+order,:)=alpha(1+order,:)/(scale(1+order)+realmin);
    
    delta(1+order,:) = alpha(1+order,:);    % Eq. 32(a) Rabiner (1989)
    % Eq. 32(b) Psi already zero
    for i=2+order:T(tr)
        alpha(i,:)=(alpha(i-1,:)*P).*B(i,:);
        scale(i)=sum(alpha(i,:));
        alpha(i,:)=alpha(i,:)/(scale(i)+realmin);
        
        for k=1:K,
            v=delta(i-1,:).*P(:,k)';
            mv=max(v);
            delta(i,k)=mv*B(i,k);  % Eq 33a Rabiner (1989)
            if length(find(v==mv)) > 1
                % no unique maximum - so pick one at random
                tmp1=find(v==mv);
                tmp2=rand(length(tmp1),1);
                [~,tmp4]=max(tmp2);
                psi(i,k)=tmp4;
            else
                psi(i,k)=find(v==mv);  % ARGMAX; Eq 33b Rabiner (1989)
            end
        end;
        
        % SCALING FOR DELTA ????
        dscale(i)=sum(delta(i,:));
        delta(i,:)=delta(i,:)/(dscale(i)+realmin);
    end;
    
    % Get beta values for single state decoding
    beta(T(tr),:)=ones(1,K)/scale(T(tr));
    for i=T(tr)-1:-1:1+order
        beta(i,:)=(beta(i+1,:).*B(i+1,:))*(P')/scale(i);
    end;
    
    xi=zeros(T(tr)-1-order,K*K);
    for i=1+order:T(tr)-1
        t=P.*( alpha(i,:)' * (beta(i+1,:).*B(i+1,:)));
        xi(i-order,:)=t(:)'/sum(t(:));
    end;
    
    delta=delta(1+order:T(tr),:);
    psi=psi(1+order:T(tr),:);
    
    % Backtracking for Viterbi decoding
    id = find(delta(T(tr)-order,:)==max(delta(T(tr)-order,:)));% Eq 34b Rabiner;
    q_star(T(tr)-order) = id(1);
    for i=T(tr)-1-order:-1:1,
        q_star(i) = psi(i+1,q_star(i+1));
    end
    
    Path( (1:(T(tr)-order)) + tacc ) = q_star;
    tacc = tacc + T(tr)-order;
    
end

end

