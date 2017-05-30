function [Path,Xi] = hmmdecode(data,T,hmm,type,residuals,reproc)
%
% State time course and Viterbi decoding for hmm
% The algorithm is run for the whole data set, including those whose class
% was fixed. This means that the assignment for those can be different.
%
% INPUT
% data          observations, either a struct with X (time series) and C (classes, optional)
%                             or just a matrix containing the time series
% T             length of series
% hmm           hmm data structure
% type          0, state time courses; 1, viterbi path
% residuals     in case we train on residuals, the value of those (optional)
%
% OUTPUT
% vpath         (T x 1) maximum likelihood state sequence (type=1) OR
% vpath         (T x K) state time courses
% Xi            joint probability of past and future states conditioned on data
%                   (empty if Viterbi path is computed) 

% Author: Diego Vidaurre, OHBA, University of Oxford

if nargin<4, type = 0; end
if nargin<5, residuals = []; end
if nargin<6, reproc = 1; end

stochastic_learn = isfield(hmm.train,'BIGNbatch') && hmm.train.BIGNbatch < length(T);
N = length(T);

if xor(iscell(data),iscell(T)), error('data and T must be cells, either both or none of them.'); end

if stochastic_learn
    if ~iscell(data)
       dat = cell(N,1); TT = cell(N,1);
       for i=1:N
          t = 1:T(i);
          dat{i} = data(t,:); TT{i} = T(i);
          try data(t,:) = []; 
          catch, error('The dimension of data does not correspond to T');
          end
       end
       if ~isempty(data)
           error('The dimension of data does not correspond to T');
       end 
       data = dat; T = TT; clear dat TT
    end
    [Path,Xi] = hmmsdecode(data,T,hmm,type);
    return
end

if iscell(T)
    for i = 1:length(T)
        if size(T{i},1)==1, T{i} = T{i}'; end
    end
    if size(T,1)==1, T = T'; end
    T = cell2mat(T);
end

if reproc % Adjust the data if necessary
    train = hmm.train;
    checkdatacell;
    data = data2struct(data,T,train);
    % Standardise data and control for ackward trials
    data = standardisedata(data,T,train.standardise);
    % Hilbert envelope
    if train.onpower
        data = rawsignal2power(data,T);
    end
    % Embedding
    if length(train.embeddedlags) > 1
        [data,T] = embeddata(data,T,train.embeddedlags);
    end
    % PCA transform
    if length(train.pca) > 1 || train.pca > 0
        if isfield(train,'A')
            data.X = bsxfun(@minus,data.X,mean(data.X)); 
            data.X = data.X * train.A;
        else
            [train.A,data.X] = highdim_pca(data.X,T,train.pca,0,0,0,options.varimax,0);
        end
        % Standardise principal components and control for ackward trials
        data = standardisedata(data,T,options.standardise_pc);
        train.ndim = size(train.A,2);
        train.S = ones(train.ndim);
        orders = formorders(train.order,train.orderoffset,train.timelag,train.exptimelag);
        train.Sind = formindexes(orders,train.S);
    end
end

if type==0
   [Path,Xi] = hsinference(data,T,hmm,residuals); 
   return
end

if isstruct(data)
    if isfield(data,'C') && ~all(isnan(data.C(:)))
        warning('Pre-specified state time courses will be ignored for Viterbi path calculation')
    end
    data = data.X;
end

Xi = [];

K = length(hmm.state);

if isempty(residuals)
    if ~isfield(hmm.train,'Sind')
        orders = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
        hmm.train.Sind = formindexes(orders,hmm.train.S);
    end
    residuals =  getresiduals(data,T,hmm.train.Sind,hmm.train.maxorder,hmm.train.order,...
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

if hmm.train.useParallel==1 && N>1
    
    % to duplicate this code is really ugly but there doesn't seem to be
    % any other way - more Matlab's fault than mine
    
    Path = cell(N,1);

    parfor tr=1:N
        
        q_star = ones(T(tr)-order,1);
        
        scale=zeros(T(tr),1);
        alpha=zeros(T(tr),K);
        beta=zeros(T(tr),K);
        
        % Initialise Viterbi bits
        delta=zeros(T(tr),K);
        psi=zeros(T(tr),K);
        
        if tr==1, t0 = 0; s0 = 0;
        else t0 = sum(T(1:tr-1)); s0 = t0 - order*(tr-1);
        end
        
        B = obslike(data(t0+1:t0+T(tr),:),hmm,residuals(s0+1:s0+T(tr)-order,:));
        B(B<realmin) = realmin;
        
        % Scaling for delta
        dscale=zeros(T(tr),1);
        
        alpha(1+order,:)=Pi(:)'.*B(1+order,:);
        scale(1+order)=sum(alpha(1+order,:));
        alpha(1+order,:)=alpha(1+order,:)/(scale(1+order));
        
        delta(1+order,:) = alpha(1+order,:);    % Eq. 32(a) Rabiner (1989)
        % Eq. 32(b) Psi already zero
        for i=2+order:T(tr)
            alpha(i,:)=(alpha(i-1,:)*P).*B(i,:);
            scale(i)=sum(alpha(i,:));
            if scale(i)<realmin, scale(i) = realmin; end
            alpha(i,:)=alpha(i,:)/(scale(i));
            
            for k=1:K
                v=delta(i-1,:).*P(:,k)';
                mv=max(v);
                delta(i,k)=mv*B(i,k);  % Eq 33a Rabiner (1989)
                fmv = find(v==mv);
                if length(fmv) > 1
                    % no unique maximum - so pick one at random
                    tmp1=fmv;
                    tmp2=rand(length(tmp1),1);
                    [~,tmp4]=max(tmp2);
                    psi(i,k)=tmp4;
                else
                    psi(i,k)=fmv;  % ARGMAX; Eq 33b Rabiner (1989)
                end
            end
            
            % SCALING FOR DELTA ????
            dscale(i)=sum(delta(i,:));
            if dscale(i)<realmin, dscale(i) = realmin; end
            delta(i,:)=delta(i,:)/(dscale(i));
        end
        
        % Get beta values for single state decoding
        beta(T(tr),:)=ones(1,K)/scale(T(tr));
        for i=T(tr)-1:-1:1+order
            beta(i,:)=(beta(i+1,:).*B(i+1,:))*(P')/scale(i);
        end
        
        xi=zeros(T(tr)-1-order,K*K);
        for i=1+order:T(tr)-1
            t=P.*( alpha(i,:)' * (beta(i+1,:).*B(i+1,:)));
            xi(i-order,:)=t(:)'/sum(t(:));
        end
        
        delta=delta(1+order:T(tr),:);
        psi=psi(1+order:T(tr),:);
        
        % Backtracking for Viterbi decoding
        id = find(delta(T(tr)-order,:)==max(delta(T(tr)-order,:)));% Eq 34b Rabiner;
        q_star(T(tr)-order) = id(1);
        for i=T(tr)-1-order:-1:1
            q_star(i) = psi(i+1,q_star(i+1));
        end
        
        Path{tr} = single(q_star);
        
    end
    
    Path = cell2mat(Path);
    
else
    
    Path = zeros(sum(T)-length(T)*order,1,'single');
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
        
        B = obslike(data(t0+1:t0+T(tr),:),hmm,residuals(s0+1:s0+T(tr)-order,:));
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
            
            for k=1:K
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
            end
            
            % SCALING FOR DELTA ????
            dscale(i)=sum(delta(i,:));
            delta(i,:)=delta(i,:)/(dscale(i)+realmin);
        end
        
        % Get beta values for single state decoding
        beta(T(tr),:)=ones(1,K)/scale(T(tr));
        for i=T(tr)-1:-1:1+order
            beta(i,:)=(beta(i+1,:).*B(i+1,:))*(P')/scale(i);
        end
        
        xi=zeros(T(tr)-1-order,K*K);
        for i=1+order:T(tr)-1
            t=P.*( alpha(i,:)' * (beta(i+1,:).*B(i+1,:)));
            xi(i-order,:)=t(:)'/sum(t(:));
        end
        
        delta=delta(1+order:T(tr),:);
        psi=psi(1+order:T(tr),:);
        
        % Backtracking for Viterbi decoding
        id = find(delta(T(tr)-order,:)==max(delta(T(tr)-order,:)));% Eq 34b Rabiner;
        q_star(T(tr)-order) = id(1);
        for i=T(tr)-1-order:-1:1
            q_star(i) = psi(i+1,q_star(i+1));
        end
        
        Path( (1:(T(tr)-order)) + tacc ) = q_star;
        tacc = tacc + T(tr)-order;
        
    end
    
end

end

