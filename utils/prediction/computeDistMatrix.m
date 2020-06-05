function [D,HMMs_dualregr] = computeDistMatrix (data,T,hmm)
%
% It performs dual regression on each element of T in order to compute a
% subject-specific HMM per subject;
% Then, it uses an approximation of the KL-divergence to compute a
% subject-by-subject distance matrix. 
%
% INPUT
% data          observations; in this case it has to be a cell, each with
%               the data for one subject
% T             length of series, also a cell. 
% hmm       	hmm struct at the group level, obtained with hmmmar() function
% 
% OUTPUT
% D             (N by N) distance matrix, with the distance between each
%               pair of subjects in "HMM space"
%  HMMs_dualregr    A cell with the estimated dual-regressed HMM models
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2020)

if ~iscell(data) || ~iscell(T), error('X and T must both be cells'); end 

N = length(data);
HMMs_dualregr = cell(N,1);

for n = 1:N
    if ischar(data{n})
        fsub = data{n};
        loadfile_sub;
    else
        X = data{n};
    end
    HMMs_dualregr{n} = hmmdual(X,T{n},hmm); %preprocessing is done here
    for k = 1:length(HMMs_dualregr{n}.state) % making it lighter
        HMMs_dualregr{n}.state(k).prior = [];
    end
end

D = NaN(N);
for n1 = 1:N-1
    for n2 = n1+1:N
        D(n1,n2) = (hmm_kl(HMMs_dualregr{n1},HMMs_dualregr{n2}) ...
            + hmm_kl(HMMs_dualregr{n2},HMMs_dualregr{n1}))/2;
        D(n2,n1) = D(n1,n2);
    end
end

end