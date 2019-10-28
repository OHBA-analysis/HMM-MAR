function [covmat,corrmat,icovmat] = getFuncConn(hmm,k,verbose)
% Get the covariance, correlation and precision matrices for state k, 
%   from the estimated model hmm
% If order==0 (Gaussian distribution), these purely represents functional
%   connectivity
% If order>0 (MAR), these refer to the covariance matrix of the residual
%
% Diego Vidaurre, OHBA, University of Oxford (2016)

if nargin < 3, verbose = 1; end

if ~isfield(hmm.state(1),'Omega')
    error('The model was defined with a unique covariance matrix'); 
end
if k>length(hmm.state)
    error('k is higher than the number of states')
end
is_diagonal = size(hmm.state(k).Omega.Gam_rate,1) == 1;
if is_diagonal
    covmat = diag( hmm.state(k).Omega.Gam_rate / (hmm.state(k).Omega.Gam_shape-1) );
    icovmat = diag( hmm.state(k).Omega.Gam_irate * (hmm.state(k).Omega.Gam_shape-1) );
else
    ndim = length(hmm.state(k).Omega.Gam_rate);
    covmat = hmm.state(k).Omega.Gam_rate / (hmm.state(k).Omega.Gam_shape-ndim-1);
    icovmat = hmm.state(k).Omega.Gam_irate * (hmm.state(k).Omega.Gam_shape-ndim-1);
end

if isfield(hmm.train,'embeddedlags') && length(hmm.train.embeddedlags) > 1 && verbose
    disp(['Because you are using options.embedded lags, the resulting matrices will ' ...
        'be (lags x regions) by (lags x regions)'])
end

if isfield(hmm.train,'A')
    A = hmm.train.A;
    covmat = A * covmat * A';
    icovmat = A * icovmat * A';
end
corrmat = corrcov(covmat,0);

end
