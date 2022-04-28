function hmm = obsupdate(Gamma,hmm,residuals,XX,XXGXX,Tfactor)
%
% Update observation model
%
% INPUT
% Gamma         p(state given X)
% hmm           hmm data structure
% residuals     in case we train on residuals, the value of those.
%
% OUTPUT
% hmm           estimated HMMMAR model
%
% Author: Diego Vidaurre, OHBA, University of Oxford

K = hmm.K; 

obs_tol = 0.00005;
obs_maxit = 1; %20;
mean_change = Inf;
obs_it = 1;
p = hmm.train.lowrank; do_HMM_pca = (p > 0);
if nargin<7, Tfactor = 1; end

if ~isfield(hmm.train,'distribution') || strcmp(hmm.train.distribution,'Gaussian')
    while mean_change>obs_tol && obs_it<=obs_maxit
        last_state = hmm.state;
        if do_HMM_pca
            hmm = updatePCAparam (hmm,sum(Gamma),XXGXX,Tfactor);
        else
            %%% W
            [hmm,XW] = updateW(hmm,Gamma,residuals,XX,XXGXX,Tfactor);  
            %%% Omega
            hmm = updateOmega(hmm,Gamma,residuals,XX,XXGXX,XW,Tfactor);
            %disp(num2str(hmm.Omega.Gam_rate / hmm.Omega.Gam_shape))
            
            %%% autoregression coefficient priors
            if (isfield(hmm.train,'V') && ~isempty(hmm.train.V))
                %%% beta - one per regression coefficient
                hmm = updateBeta(hmm);
            else
                %%% sigma - channel x channel coefficients
                hmm = updateSigma(hmm);
                %%% alpha - one per order
                hmm = updateAlpha(hmm);
            end
        end

        %%% termination conditions
        obs_it = obs_it + 1;
        mean_changew = 0;
        for k = 1:K
            mean_changew = mean_changew + ...
                sum(sum(abs(last_state(k).W.Mu_W - hmm.state(k).W.Mu_W))) / numel(hmm.state(k).W.Mu_W) / K;
        end
        mean_change = mean_changew;
    end
elseif strcmp(hmm.train.distribution,'logistic')
    if isfield(hmm,'psi')
        hmm = rmfield(hmm,'psi');
    end
    while mean_change>obs_tol && obs_it<=obs_maxit
        last_state = hmm.state;
        for iY = 1:hmm.train.logisticYdim
            hmm_marginalised = logisticMarginaliseHMM(hmm,iY);
            xdim = hmm_marginalised.train.ndim-1;
            %%% W
            [hmm_temp,~] = ...
                updateW(hmm_marginalised,Gamma,residuals(:,iY),XX(:,[1:xdim,xdim+iY]),XXGXX,Tfactor);
        
            %%% and hyperparameters alpha
            hmm_temp = updateAlpha(hmm_temp);
            
            hmm = logisticMergeHMM(hmm_temp,hmm,iY);
        end
        %%% termination conditions
        mean_changew = 0;
        for k=1:K
            mean_changew = mean_changew + ...
                sum(sum(abs(last_state(k).W.Mu_W - hmm.state(k).W.Mu_W))) / numel(hmm.state(k).W.Mu_W) / K;
        end
        mean_change = mean_changew;
        obs_it = obs_it + 1;
    end
else
    hmm = updateW(hmm,Gamma,residuals,XX,XXGXX);
end
end
