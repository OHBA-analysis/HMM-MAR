function [hmm] = obsupdate (X,T,Gamma,hmm,residuals)
%
% Update observation model
%
% INPUT
% X             observations
% T             length of series
% Gamma         p(state given X)
% hmm           hmm data structure
% residuals     in case we train on residuals, the value of those.
%
% OUTPUT
% hmm           estimated HMMMAR model
%
% Author: Diego Vidaurre, OHBA, University of Oxford

K=hmm.K;

obs_tol = 0.00005;
obs_maxit = 1; %20;
mean_change = Inf;
obs_it = 1;

% Some stuff that will be later used
Gammasum = sum(Gamma);
XXGXX = cell(K,1);
setxx;
Tres = sum(T) - length(T)*hmm.train.maxorder;

while mean_change>obs_tol && obs_it<=obs_maxit,
    
    last_state = hmm.state;
        
    %%% W
    [hmm,XW] = updateW(hmm,Gamma,residuals,XX,XXGXX);

    %%% Omega
    hmm = updateOmega(hmm,Gamma,Gammasum,residuals,Tres,XX,XXGXX,XW);
    
    %%% sigma - channel x channel coefficients
    hmm = updateSigma(hmm);
    
    %%% alpha - one per order
    hmm = updateAlpha(hmm);
    
    %%% termination conditions
    obs_it = obs_it + 1;
    mean_changew = 0;
    for k=1:K
        mean_changew = mean_changew + sum(sum(abs(last_state(k).W.Mu_W - hmm.state(k).W.Mu_W))) / length(orders) / sum(hmm.train.S(:)) / K;
    end;
    mean_change = mean_changew;
end;

end
