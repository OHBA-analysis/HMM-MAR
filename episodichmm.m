function [ehmm, Gamma, GammaInit, crithist] = episodichmm (data,T,options)
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
% ehmm           estimated ehmm model
% Gamma         estimated p(state | data)
% GammaInit     The HMM-initialised Gamma that is fed to the ehmm inference
%
% Author: Diego Vidaurre, 
%         CFIN, Aarhus University / OHBA, University of Oxford (2021)

options.episodic = true;
[ehmm, Gamma, ~, ~, GammaInit, ~, crithist] = hmmmar(data,T,options);

end