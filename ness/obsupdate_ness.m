function ness = obsupdate_ness(T,Gamma,ness,residuals,XX,Tfactor)
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

if isfield(ness.train,'distribution') && strcmp(ness.train.distribution,'logistic')
    error('Logistic regression not yet implemented for NESS')
end

ness = updateW_ness(ness,Gamma,residuals,XX,Tfactor);

% Omega
ness = updateOmega_ness(ness,Gamma,residuals,T,XX,Tfactor);

% autoregression coefficient priors
ness = updateSigma_ness(ness); % sigma - channel x channel coefficients
ness = updateAlpha_ness(ness); % alpha - one per order

end
