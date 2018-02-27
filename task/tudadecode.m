function [Gamma,vpath,Xi] = tudadecode(X,Y,T,tuda,options)
% Having estimated the TUDA model (i.e. the corresponding decoding models)
% in the same or a different data set, this function finds the model time
% courses (with no re-estimation of the decoding parameters) 
%
% INPUT
% X: Brain data, (time by regions)
% Y: Stimulus, (time by q); q is no. of stimulus features
% T: Length of series
% tuda: Estimated TUDA model, using tudatrain
% options: structure with the training options - see documentation in 
%                       https://github.com/OHBA-analysis/HMM-MAR/wiki
%           (It should be the same than when tudatrain was run) 
%
% OUTPUT 
% Gamma: Time courses of the states (decoding models) probabilities given data
% vpath: Most likely state path of hard assignments
% Xi            joint probability of past and future states conditioned on data
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2018)

N = length(T); 

% Check options and put data in the right format
[X,Y,T,options,stats.R2_pca] = preproc4hmm(X,Y,T,options); 
parallel_trials = options.parallel_trials; 
options = rmfield(options,'parallel_trials');
if isfield(options,'add_noise'), options = rmfield(options,'add_noise'); end
p = size(X,2); q = size(Y,2);

% Put X and Y together
Ttmp = T;
T = T + 1;
Z = zeros(sum(T),q+p,'single');
for n=1:N
    t1 = (1:T(n)) + sum(T(1:n-1));
    t2 = (1:Ttmp(n)) + sum(Ttmp(1:n-1));
    Z(t1(1:end-1),1:p) = X(t2,:);
    Z(t1(2:end),(p+1):end) = Y(t2,:);
end 

% Run TUDA inference
options.S = -ones(p+q);
options.S(1:p,p+1:end) = 1;
options.updateObs = 0;
options.updateGamma = 1;
options.hmm = tuda; 
options.repetitions = 0;
[~,Gamma,Xi,vpath] = hmmmar(Z,T,options);

end
