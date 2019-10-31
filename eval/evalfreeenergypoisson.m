function [FrEn,avLL] = evalfreeenergypoisson(T,Gamma,Xi,hmm,residuals,XX,todo)
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
ndim = size(hmm.state(1).W.W_shape,2);

if isfield(hmm.train,'B'), Q = size(hmm.train.B,2);
else, Q = ndim; end
pcapred = hmm.train.pcapred>0;
if pcapred, M = hmm.train.pcapred; end

Tres = sum(T) - length(T)*hmm.train.maxorder;
S = hmm.train.S==1;
regressed = sum(S,1)>0;

X=residuals;
Xdim = ndim;

if (nargin<6 || isempty(residuals)) && todo(2)==1
    if ~isfield(hmm.train,'Sind')
        orders = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
        hmm.train.Sind = formindexes(orders,hmm.train.S); 
    end
    if ~hmm.train.zeromean, hmm.train.Sind = [true(1,ndim); hmm.train.Sind]; end
    residuals =  getresiduals(X,T,hmm.train.Sind,hmm.train.maxorder,hmm.train.order,...
        hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag,hmm.train.zeromean);
end

%Entropy of Gamma 
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
KLdiv = 0;
n=Xdim+1;
if todo(5)==1
    W_KL=0;
    for k=1:K
        hs=hmm.state(k);
        pr=hmm.state(k).prior;
        for xd=1:Xdim
            W_KL = W_KL + sum(gamma_kl(hs.W.W_shape(xd),pr.alpha.Gam_shape, ...
                hs.W.W_rate(1),pr.alpha.Gam_rate));
        end
    end
    KLdiv = KLdiv + W_KL;      
end

% data log likelihood:
if isfield(hmm,'Gamma');hmm=rmfield(hmm,'Gamma');end
hmm.Gamma = Gamma;

%exp_H_LL = PoissonLogLikelihood(Y,X,hmm);

L = zeros(sum(T),K);  
constterm = -gammaln(X+1);
for k=1:K
    % note the expectation of log(lambda) is -log(lambda_b) + psigamma(lambda_a)
    num = (X.*(repmat(psi(hmm.state(k).W.W_shape),sum(T),1)-log(hmm.state(k).W.W_rate)))- hmm.state(k).W.W_mean;
    num = num.*repmat(Gamma(:,k),1,Xdim);
    L(:,k) = sum(num+constterm,2);
end

avLL=sum(L(:));

FrEn=[-Entr -avLL -avLLGamma +KLdivTran +KLdiv];




end