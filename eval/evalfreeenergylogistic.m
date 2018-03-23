function [FrEn,avLL] = evalfreeenergylogistic(T,Gamma,Xi,hmm,residuals,XX,todo)
% Computes the Free Energy of an HMM depending on observation model
%
% INPUT
% X            observations
% T            length of series
% Gamma        probability of states conditioned on data
% Xi           joint probability of past and future states conditioned on data
% hmm          hmm structure
% residuals    in case we train on residuals, the value of those.
%
% OUTPUT
% FrEn         the variational free energy, separated in different terms:
%                   element 1: data negative loglikelihood
%                   element 2: KL for initial and transition probabilities
%                   elements 3: KL for the state parameters
% avLL         log likelihood of the observed data, per trial
%
% Author: Diego Vidaurre, OHBA, University of Oxford

if nargin<8, todo = ones(1,5); end

K = length(hmm.state);
if (nargin<7 || isempty(XX)) && todo(2)==1
    setxx; % build XX and get orders
end
if isfield(hmm.state(1),'W')
    ndim = size(hmm.state(1).W.Mu_W,2);
else
    ndim = size(hmm.state(1).Omega.Gam_rate,2);
end
if isfield(hmm.train,'B'), Q = size(hmm.train.B,2);
else, Q = ndim; end
pcapred = hmm.train.pcapred>0;
if pcapred, M = hmm.train.pcapred; end

Tres = sum(T) - length(T)*hmm.train.maxorder;
S = hmm.train.S==1;
regressed = sum(S,1)>0;
% ltpi = sum(regressed)/2 * log(2*pi);

 % Set Y (unidimensional for now) and X: 
Xdim = size(XX,2)-hmm.train.logisticYdim;
X=XX(:,1:Xdim);
Y=XX(2:end,(Xdim+1):end);
Y((end+1),1)=Y(end,1);
% HACK FOR NOW - set trial boundaries to 1, not zero:
Y(Y==0)=1;
T=size(X,1);

if (nargin<6 || isempty(residuals)) && todo(2)==1
    if ~isfield(hmm.train,'Sind')
        orders = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
        hmm.train.Sind = formindexes(orders,hmm.train.S); 
    end
    if ~hmm.train.zeromean, hmm.train.Sind = [true(1,ndim); hmm.train.Sind]; end
    residuals =  getresiduals(X,T,hmm.train.Sind,hmm.train.maxorder,hmm.train.order,...
        hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag,hmm.train.zeromean);
end

% Entropy of Gamma - NO NOT USED IN LOGISTIC MODEL
% Entr = [];
% if todo(1)==1
%     Entr = GammaEntropy(Gamma,Xi,T,hmm.train.maxorder);
% end

% % loglikelihood of Gamma
% avLLGamma = [];
% if todo(3)==1
%     avLLGamma = GammaavLL(hmm,Gamma,Xi,T);
% end

% P and Pi KL
KLdivTran = [];
if todo(4)==1
    KLdivTran = KLtransition(hmm);
end

% state KL
KLdiv = [];
n=Xdim+1;
if todo(5)==1
    % note for now we use arbitrary priors!!! this to be entered properly
    % with hyperparams (in due course)
    W_mu0=zeros(Xdim,1);
    W_sig0=eye(Xdim);
    
    for k=1:K
        KLdiv = [ KLdiv, gauss_kl(hmm.state(k).W.Mu_W(S),W_mu0, ...
            squeeze(hmm.state(k).W.S_W(n,S(:,n),S(:,n))),W_sig0)];
    end
      
end

% data log likelihood:
for t=1:T
    exp_H_LL(t) = loglikelihoodofH(Y(t),X(t,:),hmm,t);
end

avLL=sum(exp_H_LL);

FrEn=[ -avLL +KLdivTran +KLdiv];

% %debugging / hack to get working, using old code (this only works to check
% %W updates, not HS states!)
% params=hmm;
% for k=1:3; 
%     params.state(k).w_mu=hmm.state(k).W.Mu_W(1:4,5);
%     params.state(k).w_sig=squeeze(hmm.state(k).W.S_W(5,1:4,1:4));
% end
% params.prior.w_mu=zeros(1,4);
% params.prior.w_sig=eye(4);
% params.gamma=hmm.Gamma;
% for t=1:T
% %     xi(t)=0;
% %     for i2=1:K
% %         xi(t)=xi(t)+(params.gamma(t,i2) * ...
% %             X(t,:)*(params.state(i2).w_sig+params.state(i2).w_mu*params.state(i2).w_mu')*X(t,:)');
% %     end
% %     xi(t)=sqrt(xi(t));
% %     exp_H_LL2(t) = bayeslogregress_loglikelihoodofH(Y(t),X(t,:),params,xi(t),t);
%     exp_H_LL2(t) = bayeslogregress_loglikelihoodofH(Y(t),X(t,:),params,hmm.psi(t),t);
% end
% FE=bayeslogregress_computefreeenergy(params,X,Y);


end