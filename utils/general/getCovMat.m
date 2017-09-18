function covmat = getCovMat(hmm,k)
% hmm is the estimated HMMMAR model
% k is the state of interest
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
end