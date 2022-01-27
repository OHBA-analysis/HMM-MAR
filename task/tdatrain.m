function [Gamma,beta] = tdatrain(X,Y,T,options)
% Performs the Temporal Decoding Approach (TDA), which, unlike TUDA, *is constrained*, 
% Th assumption is that there are fewer decoders than time points (K),
% that the same decoding is active at the same time point at all trials. 
% 
% INPUT
% X: Brain data, (time by regions) or (time by trials by regions)
% Y: Stimulus, (time by q); q is no. of stimulus features
%               For binary classification problems, Y is (time by 1) and
%               has values -1 or 1
%               For multiclass classification problems, Y is (time by classes) 
%               with indicators values taking 0 or 1. 
%           If the stimulus is the same for all trials, Y can have as many
%           rows as trials, e.g. (trials by q) 
% T: Length of series or trials
% options: structure with the training options - see documentation in 
%                       https://github.com/OHBA-analysis/HMM-MAR/wiki
%
% OUTPUT
% Gamma: Time courses of the states (decoding models) probabilities given data
% beta: Estimated decoding coefficients
%
% Author: Diego Vidaurre, OHBA, University of Oxford  

% Check options and put data in the right format
[X,Y,T,options] = preproc4hmm(X,Y,T,options);
% init HMM, only if trials are temporally related
[Gamma,beta] = cluster_decoding(X,Y,T,options.K,'hmm');

end

    