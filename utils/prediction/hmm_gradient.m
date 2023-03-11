function [LL,feat] = hmm_gradient(X, hmm, options)
% [LL, feat] = hmm_gradient(X, hmm, options)
%
% computes the gradient of HMM log-likelihood for time series X (single
% subject/session) with respect to specified parameters
% for use with HMM-MAR toolbox (https://github.com/OHBA-analysis/HMM-MAR)
% 
% INPUT:
% X:            example data (timeseries of a single subject/session, in
%               format timepoints x regions)
% hmm:          HMM structure from hmmmar call, should contain requested
%               parameters as fields 
% options:      structure specifying which parameters to use
% + Pi:         include state probabilities? (true or false, default to true)
% + P:          include transition probabilities? (true or false, default to
%               true)
% + mu:         include state means (only if HMMs were estimated with mean)?
%               (true or false, default to false)
% + sigma:      include state covariances? (true or false, default to false)
% + type:       which type of features to compute, one of either 'Fisher'
%               for Fisher kernel, 'naive' for naive kernel, or
%               'naive_norm' for naive normalised kernel
% 
% OUTPUT:
% LL:           negative log likelihood
% feat:         features (e.g. gradient with respect to requested
%               parameters)
% 
% (adapted from: Laurens van der Maaten, 2010, University of California,
% San Diego)
% 
% Christine Ahrends, Aarhus University, 2022


% set options (which parameters to include)
if nargin < 2
    error('Cannot compute gradient. X and hmm must be specified')
end
if nargin < 3
    options = struct();
end

if ~isfield(options,'Pi') || isempty(options.Pi)
    include_Pi = true;
else
    include_Pi = options.Pi;
end
if ~isfield(options,'P') || isempty(options.P)
    include_P = true;
else
    include_P = options.P;
end
if ~isfield(options,'mu') || isempty(options.mu)
    include_mu = false;
else
    include_mu = options.mu;
end
if ~isfield(options,'sigma') || isempty(options.sigma)
    include_sigma = false;
else
    include_sigma = options.sigma;
end
if ~isfield(options, 'type')
    type = 'Fisher';
else
    type = options.type;
end

T = size(X,1); 
K = numel(hmm.Pi');
N = hmm.train.ndim; 

if include_Pi
    Pi = hmm.Pi';
end
if include_P
    P = hmm.P;
end

mu = zeros(N,K);
if include_mu && hmm.train.zeromean==0
    for k = 1:K
        mu(:,k) = getMean(hmm,k,false);
    end
end

sigma = zeros(N,N,K);
if include_sigma  
    for k = 1:K
        sigma(:,:,k) = getFuncConn(hmm,k,false);
    end  
end

% dual estimation to get subject-specific HMM, Gamma, Xi, likelihood, and
% transformed data (in case of embeddings)
[hmm_sub, gamma_tmp, ~, Xi_tmp, LL,Xt] = hmmdual(X,T,hmm); % embed data within hmmdual
LL = -sum(LL);

% compute gradient
if nargout > 1 && strcmp(type,'Fisher')
    gamma = gamma_tmp';
    Xi_tmp = squeeze(mean(Xi_tmp,1));
    Xi_tmp = bsxfun(@rdivide, Xi_tmp, max(sum(Xi_tmp,2), realmin));
    
    % gradient with respect to state prior
    if include_Pi
        dPi = gamma(:,1) ./ max(Pi, realmin);
    end
    
    % gradient with respect to transition probabilities
    if include_P
        dP = Xi_tmp ./ max(P, realmin);
    end
    
    % gradient with respect to state means and covariances
    if include_mu || include_sigma
        dMu = zeros(N, K);
        invSigma = zeros(size(sigma));
        for k=1:K
            invSigma(:,:,k) = inv(sigma(:,:,k));
            if include_mu
                dMu(:,k) = sum(bsxfun(@times, gamma(k,:), invSigma(:,:,k) * bsxfun(@minus, Xt.X', mu(:,k))), 2);
            end
        end
    end
    
    % gradient with respect to state covariances
    if include_sigma && strcmpi(hmm.train.distribution, 'Gaussian')
        dSigma = zeros(N, N, K);
        for k=1:K
            Xi_V = bsxfun(@minus, Xt.X', mu(:,k));
            dSigma(:,:,k) = -sum(gamma(k,:) / 2) .* invSigma(:,:,k) + ...
                .5 * invSigma(:,:,k) * (bsxfun(@times, gamma(k,:), Xi_V) * Xi_V') * invSigma(:,:,k);
        end
    end
    
    % concatenate gradients into feature matrix
    feat = [];
    if include_Pi
        feat = [feat; -dPi(:)];
    end
    if include_P
        feat = [feat; -dP(:)];
    end
    if include_mu
        feat = [feat; -dMu(:)];
    end
    if include_sigma && strcmpi(hmm.train.distribution, 'Gaussian')
        feat = [feat; -dSigma(:)];
    end
elseif nargout > 1 && (strcmp(type, 'naive') || strcmp(type, 'naive_norm'))%if gradient was not requested, vectorise parameters from dual estimation instead (to construct naive kernel)
    feat = [];
    if include_Pi
        feat = [feat; hmm_sub.Pi(:)];
    end
    if include_P
        feat = [feat; hmm_sub.P(:)];
    end
    if include_mu
        mu_sub = zeros(N,K);
        for k = 1:K
            mu_sub(:,k) = getMean(hmm_sub,k,false);
        end
        feat = [feat; mu_sub(:)];
    end
    if include_sigma && strcmpi(hmm_sub.train.distribution, 'Gaussian')
        sigma_sub = zeros(N,N,K);
        for k = 1:K
            sigma_sub(:,:,k) = getFuncConn(hmm_sub,k,false);
        end
        feat = [feat; sigma_sub(:)];
    end
    if strcmp(type, 'naive_norm')
        feat = (feat-mean(feat,1))./std(feat,1);
    end
end
end
