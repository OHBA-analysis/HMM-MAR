function L = obslikelogistic (X,hmm,residuals,XX,slicepoints)
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
% Author: Cam Higgins, OHBA, University of Oxford


K = hmm.K;
Ydim = hmm.train.logisticYdim;
if nargin<4 || size(XX,1)==0
    [T,ndim] = size(X);
    setxx; % build XX and get orders
else
    [T,ndim] = size(residuals);
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
%     Y =  getresiduals(X,T,hmm.train.S,hmm.train.maxorder,hmm.train.order,...
%         hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag,hmm.train.zeromean);
% end
% 
% Tres = T-hmm.train.maxorder;
% S = hmm.train.S==1; 
% regressed = sum(S,1)>0;
%ltpi = sum(regressed)/2 * log(2*pi);


%separate X and Y:
Xdim = size(XX,2)-hmm.train.logisticYdim;
X = XX(:,1:Xdim);
Y = residuals;
T = size(X,1);

%indices of coefficients:
S = hmm.train.S==1;
Sind = hmm.train.S==1; 
%setstateoptions;

validdimensions = find(~any(Y==0));
L = zeros(T,K,Ydim);  
for iY = validdimensions
    n= Xdim+iY;
    % determine psi:
    for k = 1:K
        WW{k,iY} = hmm.state(k).W.Mu_W(Sind(:,n),n)*hmm.state(k).W.Mu_W(Sind(:,n),n)' + ...
                            squeeze(hmm.state(k).W.S_W(n,S(:,n),S(:,n)));
    end
     
    psi = hmm.psi(slicepoints,iY);

    
    %first component constant over different states k:
    logsigpsi = log(log_sigmoid(psi));
    % implement update equations for logistic regression:
    lambdafunc = @(psi_t) ((2*psi_t).^-1).*(log_sigmoid(psi_t)-0.5);
    lambdasig = lambdafunc(psi); 

    for k = 1:K

        % determine first order component:
        comp1 = repmat(Y(:,iY),1,Xdim) .* X * hmm.state(k).W.Mu_W(Sind(:,n),n) - psi;

    %     %full timepoint by timepoint calc for comparison:
%         comp1test=zeros(1,T);
%         for t=1:T
%             comp1test(t)=Y(t,iY)* X(t,:) * hmm.state(k).W.Mu_W(Sind(:,n),n) - psi(t);
%         end

        %determine second order component
        comp2 = sum((X * WW{k,iY}) .* X , 2) - psi.^2;

        % full timepoint by timepoint calc for comparison:
    %     comp2test=zeros(1,T);
    %     for t=1:T
    %         comp2test(t)=X(t,:)*WW{k}*X(t,:)' - hmm.psi(t).^2;
    %     end

        L(1:T,k,iY) = logsigpsi +  0.5 * comp1 - lambdasig .* comp2;
    end
end
L = sum(L,3);
L = exp(L);
end
