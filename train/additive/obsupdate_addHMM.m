function hmm = obsupdate_addHMM(T,Gamma,hmm,residuals,XX,Tfactor)
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

% Some stuff that will be later used
if nargin<6, Tfactor = 1; end

if isfield(hmm.train,'distribution') && strcmp(hmm.train.distribution,'logistic')
    error('Logistic regression not yet implemented for the additiveHMM')
end

hmm = updateW_addHMM(hmm,Gamma,residuals,XX,Tfactor);

%%% Omega
hmm = updateOmega_addHMM(hmm,Gamma,residuals,T,XX,Tfactor);

%         %%% autoregression coefficient priors
%             %%% sigma - channel x channel coefficients
%             hmm = updateSigma_addHMM(hmm);
%             %%% alpha - one per order
%             hmm = updateAlpha_addHMM(hmm);
%         end

end
