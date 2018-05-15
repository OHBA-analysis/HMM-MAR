function L = obslikelogistic (X,hmm,residuals,XX,cache,slicepoints)
%
% Evaluate likelihood of data given observation model, for one continuous trial
%
% INPUT
% X          N by ndim data matrix
% hmm        hmm data structure
% Y         for now, Y is stored in last column of XX.
% XX        alternatively to X (which in this case can be specified as []),
%               XX can be provided as computed by setxx.m
% OUTPUT
% B          Likelihood of N data points
%
% Author: Diego Vidaurre, OHBA, University of Oxford


% not familiar with caching commands so omitting for now
% if nargin < 5 || isempty(cache) 
%     use_cache = false;
% else
%     use_cache = true;
% end

K=hmm.K;
if nargin<4 || size(XX,1)==0
    [T,ndim]=size(X);
    setxx; % build XX and get orders
else
    [T,ndim] = size(residuals);
%    T = T + hmm.train.maxorder;
end
if nargin==6
    Gamma=hmm.Gamma(slicepoints,:);
    if isfield(hmm,'psi');
    psi=hmm.psi(slicepoints);
    hmm=rmfield(hmm,'psi');
    hmm.psi=psi;
    end
end
    
% if isfield(hmm.train,'B'), Q = size(hmm.train.B,2);
% else Q = ndim; end
% 
% if nargin<3 || isempty(Y)
%     ndim = size(X,2);
%     if ~isfield(hmm.train,'Sind')
%         if hmm.train.pcapred==0
%             orders = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
%             hmm.train.Sind = formindexes(orders,hmm.train.S);
%         else
%            hmm.train.Sind = ones(hmm.train.pcapred,ndim); 
%         end
%     end
%     if ~hmm.train.zeromean, hmm.train.Sind = [true(1,ndim); hmm.train.Sind]; end
%     Y =  getresiduals(X,T,hmm.train.Sind,hmm.train.maxorder,hmm.train.order,...
%         hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag,hmm.train.zeromean);
% end
% 
% Tres = T-hmm.train.maxorder;
% S = hmm.train.S==1; 
% regressed = sum(S,1)>0;
%ltpi = sum(regressed)/2 * log(2*pi);
L = zeros(T,K);  

%separate X and Y:
Xdim = size(XX,2)-hmm.train.logisticYdim;
X=XX(:,1:Xdim);
% Y=XX(2:end,(Xdim+1):end);
% Y((end+1),1)=Y(end,1);
Y=residuals;
T=size(X,1);

%indices of coefficients:
S = hmm.train.S==1;
Sind = hmm.train.S==1; 
%setstateoptions;

%for Y with >2 dimensions, change here!!!
n=Xdim+hmm.train.logisticYdim;

% determine psi:
for k=1:K
    WW{k}=hmm.state(k).W.Mu_W(Sind(:,n),n)*hmm.state(k).W.Mu_W(Sind(:,n),n)' + ...
                            squeeze(hmm.state(k).W.S_W(n,S(:,n),S(:,n)));
end

if ~isfield(hmm,'psi') 
    hmm.psi=zeros(1,T);
    for t=1:T
        for i=1:K
            gamWW(:,:,i) = Gamma(t,i)* WW{i};  
        end
        hmm.psi(1,t)=sqrt(X(t,:) * sum(gamWW,3) * X(t,:)');
        if mod(t,100)==0;fprintf(['\n',int2str(t)]);end
    end
end
if size(hmm.psi,2)>1
    hmm.psi=hmm.psi';
end

%first component constant over different states k:
logsigpsi=log(logsig(hmm.psi));
% implement update equations for logistic regression:
lambdafunc = @(psi_t) ((2*psi_t).^-1).*(logsig(psi_t)-0.5);
lambdasig = lambdafunc(hmm.psi); 

for k=1:K
    
    % determine first order component:
    comp1 = repmat(Y,1,Xdim) .* X * hmm.state(k).W.Mu_W(Sind(:,n),n) - hmm.psi;
    
%     %full timepoint by timepoint calc for comparison:
    comp1test=zeros(1,T);
    for t=1:T
        comp1test(t)=Y(t)* X(t,:) * hmm.state(k).W.Mu_W(Sind(:,n),n) - hmm.psi(t);
    end
    
    %determine second order component
    comp2 = sum((X * WW{k}) .* X , 2) - hmm.psi.^2;
    
    % full timepoint by timepoint calc for comparison:
%     comp2test=zeros(1,T);
%     for t=1:T
%         comp2test(t)=X(t,:)*WW{k}*X(t,:)' - hmm.psi(t).^2;
%     end
    
    L(1:T,k)= logsigpsi +  0.5 * comp1 - lambdasig .* comp2;
end
L=exp(L);
end
