function hmm = obsupdate_addHMM(T,Gamma,hmm,residuals,XX,XXGXX,Tfactor)
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

K = hmm.K;
ndim = size(residuals,2);

% Some stuff that will be later used
if nargin<7, Tfactor = 1; end
Gamma = [Gamma sum(1 - Gamma,2) ];
Gammasum = sum(Gamma);

update_residuals = hmm.train.order > 0 || hmm.train.zeromean == 0;

if update_residuals
    meand = computeStateResponses(XX,ndim,hmm,Gamma,1:K,true);
else
    residuals_k = residuals;
end

if ~isfield(hmm.train,'distribution') || ~strcmp(hmm.train.distribution,'logistic')
    
    for k = randperm(K)

        if update_residuals
            meand_k = computeStateResponses(XX,ndim,hmm,Gamma,k,true);
            meand = meand - meand_k;
            residuals_k = residuals - meand;
        end
        
        %%% W
        [hmm,XW] = updateW(hmm,Gamma,residuals_k,XX,XXGXX,Tfactor,k);
        %%% Omega
        hmm = updateOmega(hmm,Gamma,Gammasum,residuals_k,T,XX,XXGXX,XW,Tfactor,k);
        %disp(num2str(hmm.Omega.Gam_rate / hmm.Omega.Gam_shape))
        
        %%% autoregression coefficient priors
        if (isfield(hmm.train,'V') && ~isempty(hmm.train.V))
            %%% beta - one per regression coefficient
            hmm = updateBeta(hmm,k);
        else
            %%% sigma - channel x channel coefficients
            hmm = updateSigma(hmm,k);
            %%% alpha - one per order
            hmm = updateAlpha(hmm,k);
        end
        
        if update_residuals
            meand_k = computeStateResponses(XX,ndim,hmm,Gamma,k,true);
            meand = meand + meand_k;
        end
        
    end
    
else
  error('Logistic regression not yet implemented for the additiveHMM')
end
end
