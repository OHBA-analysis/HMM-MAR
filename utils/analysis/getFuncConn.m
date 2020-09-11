function [covmat,corrmat,icovmat] = getFuncConn(hmm,k,verbose)
% Get the covariance, correlation and precision matrices for state k, 
%   from the estimated model hmm
% If order==0 (Gaussian distribution), these purely represents functional
%   connectivity
% If order>0 (MAR), these refer to the covariance matrix of the residual
%
% Diego Vidaurre, OHBA, University of Oxford (2016)

if nargin < 3, verbose = 1; end
p = hmm.train.lowrank; do_HMM_pca = (p > 0);

if ~isfield(hmm.state(1),'Omega') && ~do_HMM_pca
    error('The model was defined with a unique covariance matrix'); 
end
if k>length(hmm.state)
    error('k is higher than the number of states')
end
is_diagonal = ~do_HMM_pca && size(hmm.state(k).Omega.Gam_rate,1) == 1;

if do_HMM_pca
    ndim = size(hmm.state(k).W.Mu_W,1);
    covmat = hmm.state(k).W.Mu_W * hmm.state(k).W.Mu_W' + ...
        hmm.Omega.Gam_rate / hmm.Omega.Gam_shape * eye(ndim); 
    icovmat = inv(covmat); 
elseif is_diagonal
    covmat = diag( hmm.state(k).Omega.Gam_rate / (hmm.state(k).Omega.Gam_shape-1) );
    if ~isfield(hmm.state(k).Omega,'Gam_irate')
        hmm.state(k).Omega.Gam_irate = 1 ./ hmm.state(k).Omega.Gam_irate;
    end
    icovmat = diag( hmm.state(k).Omega.Gam_irate * (hmm.state(k).Omega.Gam_shape-1) );
else
    ndim = length(hmm.state(k).Omega.Gam_rate);
    covmat = hmm.state(k).Omega.Gam_rate / (hmm.state(k).Omega.Gam_shape-ndim-1);
    if ~isfield(hmm.state(k).Omega,'Gam_irate')
        hmm.state(k).Omega.Gam_irate = inv(hmm.state(k).Omega.Gam_rate);
    end
    icovmat = hmm.state(k).Omega.Gam_irate * (hmm.state(k).Omega.Gam_shape-ndim-1);
end

if isfield(hmm.train,'embeddedlags') && length(hmm.train.embeddedlags) > 1 && verbose
    disp(['Because you are using options.embedded lags, the resulting matrices will ' ...
        'be (lags x regions) by (lags x regions)'])
end

if isfield(hmm.train,'A') 
    if do_HMM_pca, error('Incorrect parametrisation'); end 
    A = hmm.train.A;
    covmat = A * covmat * A';
    icovmat = A * icovmat * A';
end
corrmat = corrcov(covmat,0);

end
