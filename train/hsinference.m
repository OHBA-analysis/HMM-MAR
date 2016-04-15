function [Gamma,Gammasum,Xi,LL,scale,B] = hsinference(data,T,hmm,residuals,options,XX)
%
% inference engine for HMMs.
%
% INPUT
%
% data      Observations - a struct with X (time series) and C (classes)
% T         Number of time points for each time series
% hmm       hmm data structure
% residuals in case we train on residuals, the value of those.
% XX        optionally, XX, as computed by setxx.m, can be supplied
%
% OUTPUT
%
% Gamma     Probability of hidden state given the data
% Gammasum  sum of Gamma over t
% Xi        joint Prob. of child and parent states given the data
% LL        Log-likelihood
%
% Author: Diego Vidaurre, OHBA, University of Oxford

N = length(T);
K = length(hmm.state);

if ~isfield(hmm,'train')
    if nargin<5 || isempty(options),
        error('You must specify the field options if hmm.train is missing');
    end
    hmm.train = checkoptions(options,data.X,T,0);
end
order = hmm.train.maxorder;

if iscell(data)
    data = cell2mat(data);
end
if ~isstruct(data),
    data = struct('X',data); 
    data.C = NaN(size(data.X,1)-order*length(T),K);
end

if nargin<4 || isempty(residuals)
    ndim = size(data.X,2);
    if ~isfield(hmm.train,'Sind'),
        orders = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
        hmm.train.Sind = formindexes(orders,hmm.train.S);
    end
    if ~hmm.train.zeromean, hmm.train.Sind = [true(1,ndim); hmm.train.Sind]; end
    residuals =  getresiduals(data.X,T,hmm.train.Sind,hmm.train.maxorder,hmm.train.order,...
        hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag,hmm.train.zeromean);
end

if ~isfield(hmm,'P')
    hmm = hmmhsinit (hmm);
end

if nargin<6 || isempty(XX)
    setxx;
end

Gamma = cell(N,1);
LL = zeros(N,1);
scale = cell(N,1);
Gammasum = zeros(N,K);
Xi = cell(N,1);
B = cell(N,1);

n_argout = nargout;

if hmm.train.useParallel==1 && N>1
            
    % to duplicate this code is really ugly but there doesn't seem to be
    % any other way - more Matlab's fault than mine 
    parfor in=1:N 
        Bt = []; sc = [];
        t0 = sum(T(1:in-1)); s0 = t0 - order*(in-1);
        if order>0
            C = [zeros(order,K); data.C(s0+1:s0+T(in)-order,:)];
            R = [zeros(order,size(residuals,2));  residuals(s0+1:s0+T(in)-order,:)];
        else
            C = data.C(s0+1:s0+T(in)-order,:);
            R = residuals(s0+1:s0+T(in)-order,:);
        end
        % we jump over the fixed parts of the chain
        t = order+1;
        xi = []; gamma = []; gammasum = zeros(1,K); ll = 0;
        while t<=T(in)
            if isnan(C(t,1)), no_c = find(~isnan(C(t:T(in),1)));
            else no_c = find(isnan(C(t:T(in),1)));
            end
            if t>order+1
                if isempty(no_c), slicer = (t-1):T(in); %slice = (t-order-1):T(in);
                else slicer = (t-1):(no_c(1)+t-2); %slice = (t-order-1):(no_c(1)+t-2);
                end;
            else
                if isempty(no_c), slicer = t:T(in); %slice = (t-order):T(in);
                else slicer = t:(no_c(1)+t-2); %slice = (t-order):(no_c(1)+t-2);
                end;
            end
            XXt = cell(length(XX),1);
            for k=1:length(XX), XXt{k} = XX{k}(slicer + s0 - order,:); end
            if isnan(C(t,1))
                [gammat,xit,Bt,sc] = nodecluster(XXt,K,hmm,R(slicer,:));
            else
                gammat = zeros(length(slicer),K);
                if t==order+1, gammat(1,:) = C(slicer(1),:); end
                xit = zeros(length(slicer)-1, K^2);
                for i=2:length(slicer)
                    gammat(i,:) = C(slicer(i),:);
                    xitr = gammat(i-1,:)' * gammat(i,:) ;
                    xit(i-1,:) = xitr(:)';
                end
                if n_argout>=4, Bt = obslike([],hmm,R(slicer,:),XXt); end
                if n_argout==5, sc = ones(length(slicer),1); end
            end
            if t>order+1,
                gammat = gammat(2:end,:);
            end
            xi = [xi; xit];
            gamma = [gamma; gammat];
            gammasum = gammasum + sum(gamma);
            if n_argout>=4, ll = ll + sum(sum(log(Bt(order+1:end,:)) .* gammat,2)); end
            if n_argout>=5, scale = [scale; sc ]; end
            if isempty(no_c), break;
            else t = no_c(1)+t-1;
            end;
        end
        Gamma{in} = gamma;
        Gammasum(in,:) = gammasum;
        if n_argout>=4, LL(in) = ll; end
        %Xi=cat(1,Xi,reshape(xi,T(in)-order-1,K,K));
        Xi{in} = reshape(xi,T(in)-order-1,K,K);
    end
    
else
    
    for in=1:N % this is exactly the same than the code above but changing parfor by for
       Bt = []; sc = [];
        t0 = sum(T(1:in-1)); s0 = t0 - order*(in-1);
        if order>0
            C = [zeros(order,K); data.C(s0+1:s0+T(in)-order,:)];
            R = [zeros(order,size(residuals,2));  residuals(s0+1:s0+T(in)-order,:)];
        else
            C = data.C(s0+1:s0+T(in)-order,:);
            R = residuals(s0+1:s0+T(in)-order,:);
        end
        % we jump over the fixed parts of the chain
        t = order+1;
        xi = []; gamma = []; gammasum = zeros(1,K); ll = 0;
        while t<=T(in)
            if isnan(C(t,1)), no_c = find(~isnan(C(t:T(in),1)));
            else no_c = find(isnan(C(t:T(in),1)));
            end
            if t>order+1
                if isempty(no_c), slicer = (t-1):T(in); %slice = (t-order-1):T(in);
                else slicer = (t-1):(no_c(1)+t-2); %slice = (t-order-1):(no_c(1)+t-2);
                end;
            else
                if isempty(no_c), slicer = t:T(in); %slice = (t-order):T(in);
                else slicer = t:(no_c(1)+t-2); %slice = (t-order):(no_c(1)+t-2);
                end;
            end
            XXt = cell(length(XX),1);
            for k=1:length(XX), XXt{k} = XX{k}(slicer + s0 - order,:); end
            if isnan(C(t,1))
                [gammat,xit,Bt,sc] = nodecluster(XXt,K,hmm,R(slicer,:));
            else
                gammat = zeros(length(slicer),K);
                if t==order+1, gammat(1,:) = C(slicer(1),:); end
                xit = zeros(length(slicer)-1, K^2);
                for i=2:length(slicer)
                    gammat(i,:) = C(slicer(i),:);
                    xitr = gammat(i-1,:)' * gammat(i,:) ;
                    xit(i-1,:) = xitr(:)';
                end
                if nargout>=4, Bt = obslike([],hmm,R(slicer,:),XXt); end
                if nargout==5, sc = ones(length(slicer),1); end
            end
            if t>order+1,
                gammat = gammat(2:end,:);
            end
            xi = [xi; xit];
            gamma = [gamma; gammat];
            gammasum = gammasum + sum(gamma);
            if nargout>=4, ll = ll + sum(sum(log(Bt(order+1:end,:)) .* gammat,2)); end
            if nargout>=5, scale = [scale; sc ]; end
            if isempty(no_c), break;
            else t = no_c(1)+t-1;
            end;
        end
        Gamma{in} = gamma;
        Gammasum(in,:) = gammasum;
        if nargout>=4, LL(in) = ll; end
        %Xi=cat(1,Xi,reshape(xi,T(in)-order-1,K,K));
        Xi{in} = reshape(xi,T(in)-order-1,K,K);
    end
end

% join
Gamma = cell2mat(Gamma);
scale = cell2mat(scale);
Xi = cell2mat(Xi);
B  = cell2mat(B);

end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [Gamma,Xi,B,scale] = nodecluster(XX,K,hmm,residuals)
% inference using normal foward backward propagation

order = hmm.train.maxorder;
T = size(residuals,1) + order;
P = hmm.P;
Pi = hmm.Pi;

B = obslike([],hmm,residuals,XX);
B(B<realmin) = realmin;

% pass to mex file?
if ( (ismac || isunix) && hmm.train.useMEX ==1 && ...
        exist('hidden_state_inference_mx', 'file') == 3 && ...
        exist('ignore_MEX', 'file') == 0 )
    finish = 1;
    try
        [Gamma, Xi, scale] = hidden_state_inference_mx(B, Pi, P, order);
    catch
        fprintf('MEX file cannot be used, going on to Matlab code..\n')
        fclose(fopen('ignore_MEX', 'w'));
        finish = 0;
    end
    if finish==1, return; end
end

scale=zeros(T,1);
alpha=zeros(T,K);
beta=zeros(T,K);

alpha(1+order,:)=Pi.*B(1+order,:);
scale(1+order)=sum(alpha(1+order,:));
alpha(1+order,:)=alpha(1+order,:)/scale(1+order);
for i=2+order:T
    alpha(i,:)=(alpha(i-1,:)*P).*B(i,:);
    scale(i)=sum(alpha(i,:));		% P(X_i | X_1 ... X_{i-1})
    alpha(i,:)=alpha(i,:)/scale(i);
end;

scale(scale<realmin) = realmin;

beta(T,:)=ones(1,K)/scale(T);
for i=T-1:-1:1+order
    beta(i,:)=(beta(i+1,:).*B(i+1,:))*(P')/scale(i);
    beta(i,beta(i,:)>realmax) = realmax;
end;
Gamma=(alpha.*beta);
Gamma=Gamma(1+order:T,:);
Gamma=rdiv(Gamma,rsum(Gamma));
 
Xi=zeros(T-1-order,K*K);
for i=1+order:T-1
    t=P.*( alpha(i,:)' * (beta(i+1,:).*B(i+1,:)));
    Xi(i-order,:)=t(:)'/sum(t(:));
end
end
