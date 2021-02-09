function [ness, Gamma, GammaInit, crithist] = nessmodel (data,T,options)
%
% Train Hidden Markov Model using using Variational Framework
%
% INPUTS
% data          observations; either a struct with X (time series) 
%                             or a matrix containing the time series,
%                             or a list of file names
% T             length of series
% options       structure with the training options - see documentation in 
%                       https://github.com/OHBA-analysis/HMM-MAR/wiki
%
% OUTPUTS
% ness           estimated NESS model
% Gamma         estimated p(state | data)
% GammaInit     The HMM-initialised Gamma that is fed to the NESS inference
%
% Author: Diego Vidaurre, 
%         CFIN, Aarhus University / OHBA, University of Oxford (2021)

options.nessmodel = true;
[ness, Gamma, ~, crithist, GammaInit] = hmmmar(data,T,options); 

end