function covmat = getCovMat(hmm,k)
% hmm is the estimated HMMMAR model
% k is the state of interest
covmat = hmm.state(k).Omega.Gam_rate / hmm.state(k).Omega.Gam_shape;
end