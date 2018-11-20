function [covmat,corrmat] = getFuncConn(hmm,k)
% Get the covariance and correlation matrices for state k, 
%   from the estimated model hmm
% If order==0 (Gaussian distribution), these purely represents functional
%   connectivity
% If order>0 (MAR), these refer to the covariance matrix of the residual
%
% Diego Vidaurre, OHBA, University of Oxford (2016)

if ~isfield(hmm.state(1),'Omega')
    error('The model was defined with a unique covariance matrix'); 
end
if k>length(hmm.state)
    error('k is higher than the number of states')
end
is_diagonal = size(hmm.state(k).Omega.Gam_rate,1) == 1;
if is_diagonal
    covmat = diag( hmm.state(k).Omega.Gam_rate / (hmm.state(k).Omega.Gam_shape-1) );
else
    ndim = length(hmm.state(k).Omega.Gam_rate);
    covmat = hmm.state(k).Omega.Gam_rate / (hmm.state(k).Omega.Gam_shape-ndim-1);
end
try
    corrmat = corrcov(covmat);
catch
    corrmat = [];
    warning(['The covariance matrix of this state is not well-defined, ' ...
        'so the correlation matrix was not computed (second output argument is empty).' ... 
        'Most probably, the inference did not assign enough time points to this state.'])end

if isfield(hmm.train,'A')
    A = hmm.train.A;
    if ~isempty(corrmat)
        corrmat = A * corrmat * A';
    end
    covmat = A * covmat * A';
end

end