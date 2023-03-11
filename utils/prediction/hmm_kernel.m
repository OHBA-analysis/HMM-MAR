function [K, feat, Dist] = hmm_kernel(X_all, hmm, options)
% [K, feat, Dist] = hmm_kernel(X_all, hmm, options)
%
% hmm_kernel computes a kernel and feature matrix from HMMs
% implemented for linear and Gaussian versions of Fisher kernel & naive
% kernels
% for use with HMM-MAR toolbox (https://github.com/OHBA-analysis/HMM-MAR)
% 
% INPUT:
% X_all:    timeseries of all subjects/sessions (cell of size samples x 1)
%           where each cell contains a matrix of timepoints x ROIs
% hmm:      group-level HMM (hmm structure, output from hmmmar)
% options:  structure containing
% + Pi:     include state probabilities? (true or false, default to true)
% + P:      include transition probabilities? (true or false, default to
%           true)
% + mu:     include state means (only if HMMs were estimated with mean)?
%           (true or false, default to false)
% + sigma:  include state covariances? (true or false, default to false)
% + type:   which type of features to compute, one of either 'Fisher'
%           for Fisher kernel, 'naive' for naive kernel, or
%           'naive_norm' for naive normalised kernel
% + kernel: which kernel to compute, one of either 'linear' or 'Gaussian'
% + normalisation:
%           (optional) how to normalise features, e.g. 'L2-norm'
%
% OUTPUT:
% K:        Kernel specified by options.type and options.kernel (matrix 
%           of size samples x samples), e.g. for options.type='Fisher' 
%           and options.kernel = 'linear', this is the linear Fisher kernel
% feat:     features from which kernel was constructed (matrix of size
%           samples x features), e.g. for options.type='Fisher', this
%           will be the gradients of the log-likelihood of each subject
%           w.r.t. to the specified parameters (i.e. Fisher scores),
%
% Christine Ahrends, Aarhus University (2022)


if isfield(options, 'normalisation') && ~isempty(options.normalisation)
    normalisation = options.normalisation;
else
    normalisation = 'none';
end

if ~isfield(options, 'kernel') || isempty(options.kernel)
    kernel = 'linear';
else
    kernel = options.kernel;
end

if strcmpi(kernel, 'Gaussian')
    if ~isfield(options, 'tau') || isempty(options.tau)
        tau = 1; % radius of Gaussian kernel, do this in CV
    else
        tau = options.tau;
    end
end

S = size(X_all,1);
K = hmm.K;
N = hmm.train.ndim;
feat = zeros(S, (options.Pi*K + options.P*K*K + options.mu*K*N + options.sigma*K*N*N));

% get features (compute gradient if requested)
for s = 1:S
    [~, feat(s,:)] = hmm_gradient(X_all{s}, hmm, options); % if options.type='vectorised', this does not compute the gradient, but simply does dual estimation and vectorises the subject-level parameters
end
%
% NOTE: features will be in embedded space (e.g. embedded lags &
% PCA space)

% construct kernel

if strcmpi(normalisation, 'L2-norm') % normalise features (e.g. L2-norm of gradients)
    feat = bsxfun(@rdivide, feat, max(sqrt(sum(feat.^2, 1)), realmin));
end

if strcmpi(kernel, 'linear')   
    K = feat*feat';
    
elseif strcmpi(kernel, 'Gaussian')
    % get norm of feature vectors
    for i =1:S
        for j = 1:S
            Dist(i,j) = sqrt(sum(abs(feat(i,:)-feat(j,:)).^2)).^2;
        end
    end
    K = exp(-Dist/(2*tau^2));
end

end
