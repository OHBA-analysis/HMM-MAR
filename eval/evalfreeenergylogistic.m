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
%                   element 1: Gamma Entropy
%                   element 2: data negative loglikelihood
%                   element 3: Gamma negative loglikelihood
%                   element 4: KL for initial and transition probabilities
%                   element 5: KL for the state parameters
% avLL         log likelihood of the observed data, per trial
%
% Author: Cam Higgins, OHBA, University of Oxford

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
% Y=XX(2:end,(Xdim+1):end);
% Y((end+1),1)=Y(end,1);

Y=residuals;
T_full=size(X,1);

if (nargin<6 || isempty(residuals)) && todo(2)==1
    if ~isfield(hmm.train,'Sind')
        orders = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
        hmm.train.Sind = formindexes(orders,hmm.train.S); 
    end
    if ~hmm.train.zeromean, hmm.train.Sind = [true(1,ndim); hmm.train.Sind]; end
    residuals =  getresiduals(X,T,hmm.train.Sind,hmm.train.maxorder,hmm.train.order,...
        hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag,hmm.train.zeromean);
end

% Entropy of Gamma 
Entr = [];
if todo(1)==1
    Entr = GammaEntropy(Gamma,Xi,T,hmm.train.maxorder);
end

% loglikelihood of Gamma
avLLGamma = [];
if todo(3)==1
    avLLGamma = GammaavLL(hmm,Gamma,Xi,T);
end

% P and Pi KL
KLdivTran = [];
if todo(4)==1
    KLdivTran = KLtransition(hmm);
end

% state KL
n=Xdim+1;
if todo(5)==1
    W_mu0=zeros(Xdim,1);
    %W_sig0=eye(Xdim);
    alphaKL=[];
    beta_KL = [];
    for k=1:K
        hs=hmm.state(k);
        pr=hmm.state(k).prior;
        for ly = n:n+hmm.train.logisticYdim-1
            W_sig0 = diag(hs.alpha.Gam_shape ./ (hs.alpha.Gam_rate(1:Xdim,ly-Xdim)));
            beta_KL = [ beta_KL, gauss_kl(hs.W.Mu_W(1:Xdim,ly),W_mu0, ...
                squeeze(hs.W.S_W(ly,1:Xdim,1:Xdim)),W_sig0)];
            alphaKL_st=zeros(Xdim,1);
            for xd=1:Xdim
                alphaKL_st(xd) = sum(gamma_kl(hs.alpha.Gam_shape,pr.alpha.Gam_shape, ...
                    hs.alpha.Gam_rate(xd,ly-Xdim),pr.alpha.Gam_rate));
            end
            alphaKL = [alphaKL,sum(alphaKL_st)];
        end
    end
    KLdiv = sum(beta_KL) + sum(alphaKL);      
end

% data log likelihood:
if isfield(hmm,'Gamma');hmm=rmfield(hmm,'Gamma');end
hmm.Gamma = Gamma;

exp_H_LL = loglikelihoodofH(Y,X,hmm);
avLL=sum(exp_H_LL);

FrEn=[-Entr -avLL -avLLGamma +KLdivTran +KLdiv];

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