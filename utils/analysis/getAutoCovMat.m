function C = getAutoCovMat(hmm,k,verbose)
% Get the autocovariance from a TDE-HMM model, for state k, 
%
% Diego Vidaurre, OHBA, University of Oxford (2019)

if nargin < 3, verbose = 1; end
C = getFuncConn(hmm,k,verbose);

end