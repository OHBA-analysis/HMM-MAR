function tests = hmmtest (Gamma,T,Tsubject,Y,options,hmm)
% Statistical testing to differenciate between groups according to how much
% time they spend on the different states, or how quickly they switch between states. 
% For statistical testing on how the probability of the states is modulated
% by a stimulus, time point by time point, use hmmtest_epoched.  
%
% More specifically, it tests fractional occupancy and switching rates against a 
% (no. trials by no. conditions) design matrix Y using permutation testing.
% Fractional occupancy is tested per state, and also aggregated across states
% using the NPC algorithm; see Winkler et al., 2016 NI; Vidaurre et al. 2018 HBM
% Switching rate is tested also for all states simultaneously. 
% Tests are done at the subject and at the group level. 
% Tests are the group level are done by
%   - Permuting between subjects, if T == Tsubject
%   - Permuting within subjects, if not (i.e. no between subject permutation)
%
% INPUTS:
%
% Gamma: state time courses, as returned by hmmmar
% T: length of time series (no. trials by 1) in number of time points
% Tsubject: length of time series per subject (no. subjects by 1); 
%       if empty it will be interpreted that there is only one subject
% Y: design matrix  (no. trials by no. conditions) 
% options: testing related options
%   .Nperm: number of permutations (default 1000)
%   .subjectlevel: run subject-level tests? 
%           (default 1 if all subjects have at least 100 trials)
%   .grouplevel: run group-level tests? 
%           (default 1, if there are at least 2 subjects)
%   .confounds: (no. trials by q) matrix of confounds that are to be
%           regressed out before doing the testing (default none)
% hmm: hmm structure
% 
% OUTPUTS:
%
% tests: a struct with fields
%   .subjectlevel: subject-level tests, with fields containing  
%           p-values for the different measures:
%       .p_fractional_occupancy: (no. states by no. subjects)  
%       .p_aggr_fractional_occupancy: (1 by no. subjects) of aggregated pvalues
%       .p_switching_rate: (1 by no. subjects) one p-value 
%               for the global rate of switching
%   .grouplevel: group-level tests, with fields
%       .p_fractional_occupancy: (no. states by 1)  
%       .p_aggr_fractional_occupancy: 1 global pvalue of aggregated pvalues
%       .p_switching_rate: 1 global p-value 
%   Either .subjectlevel or .grouplevel will be structs with empty fields
%       if the corresponding level of testing is not run. 
%  
% Note that this p-values are not corrected for multiple comparisons
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2017)

% Put T,Tsubject in the format
if iscell(T)
    if size(T,1)==1, T = T'; end
    T = cell2mat(T);
end
if iscell(Tsubject)
    if size(Tsubject,1)==1, Tsubject = Tsubject'; end
    Tsubject = cell2mat(Tsubject);
end
if size(T,1)==1, T = T'; end
if isempty(Tsubject), Tsubject = sum(T); end  % One subject
if size(Tsubject,1)==1, Tsubject = Tsubject'; end
if sum(T) ~= sum(Tsubject), error('sum(T) must equal to sum(Tsubject)'); end
N = length(T); Nsubj = length(Tsubject);
K = size(Gamma,2); % no. of states

% Get number of trials per subject
if Nsubj > 1
    Ntrials_per_subject = zeros(Nsubj,1);
    TperSubj = cell(Nsubj,1);
    i = 1; l = 0; s = 0; j0 = 1; 
    for j = 1:N
        s = s + T(j); l = l + 1;  
        if s==Tsubject(i)
            Ntrials_per_subject(i) = l;
            TperSubj{i} = T(j0:j);
            s = 0; l = 0; i = i + 1;
            j0 = j + 1; 
        end
    end
else
    Ntrials_per_subject = N;
end

% Check options
if ~isfield(options,'Nperm'), Nperm = 1000; 
else, Nperm = options.Nperm;
end
if isfield(options,'subjectlevel'), subjectlevel = options.subjectlevel;
else, subjectlevel = (min(Ntrials_per_subject) > 100);
end
if subjectlevel && min(Ntrials_per_subject)<10
    warning('Not enough trials per subject in order to get subject-specific p-values ')
    subjectlevel = 0;
end
if isfield(options,'grouplevel'), grouplevel = options.grouplevel;
else, grouplevel = (Nsubj > 2); 
end
if ~isfield(options,'confounds'), confounds = [];
else, confounds = options.confounds;
end
if ~isfield(hmm.train,'downsample') || ~isfield(hmm.train,'Fs')
    r = 1; 
else
    r = hmm.train.downsample/hmm.train.Fs;
end

% Adjust dimensions of T and Tsubject
if isfield(hmm.train,'order') && hmm.train.order > 0
    Tsubject = ceil(r * Tsubject);
    Tsubject = Tsubject - Ntrials_per_subject * hmm.train.order; 
elseif isfield(hmm.train,'embeddedlags') && length(hmm.train.embeddedlags) > 1
    d1 = -min(0,hmm.train.embeddedlags(1));
    d2 = max(0,hmm.train.embeddedlags(end));
    Tsubject = Tsubject - Ntrials_per_subject * (d1+d2); 
    Tsubject = ceil(r * Tsubject);
end

tests = struct();
tests.subjectlevel = struct();
tests.grouplevel = struct();

% Testing at the subject level
if subjectlevel
    tests.subjectlevel.p_fractional_occupancy = zeros(K,Nsubj);
    tests.subjectlevel.p_aggr_fractional_occupancy = zeros(1,Nsubj);
    tests.subjectlevel.p_switching_rate = zeros(1,Nsubj);  
    for j = 1:Nsubj
        t = (1:Tsubject(j)) + sum(Tsubject(1:j-1));
        g = Gamma(t,:);
        jj = (1:Ntrials_per_subject(j)) + sum(Ntrials_per_subject(1:j-1));
        Yj = Y(jj,:);
        fo = getFractionalOccupancy(g,TperSubj{j},hmm.train);
        sr = getSwitchingRate(g,TperSubj{j},hmm.train);
        tests.subjectlevel.p_fractional_occupancy(:,j) = permtest_aux(fo,Yj,Nperm,confounds);
        tests.subjectlevel.p_aggr_fractional_occupancy(j) = permtest_aux_NPC(fo,Yj,Nperm,confounds);
        tests.subjectlevel.p_switching_rate(j) = permtest_aux(sr,Yj,Nperm,confounds);
    end
end
   
% Testing at the group level
if grouplevel
    if (length(T)==length(Tsubject)) 
        Ntrials_per_subject = [];
    end
    fo = getFractionalOccupancy(Gamma,T,hmm.train);
    sr = getSwitchingRate(Gamma,T,hmm.train);
    tests.grouplevel.p_fractional_occupancy = permtest_aux(fo,Y,Nperm,confounds);
    tests.grouplevel.p_aggr_fractional_occupancy = permtest_aux_NPC(fo,Y,Nperm,confounds);
    tests.grouplevel.p_switching_rate = permtest_aux(sr,Y,Nperm,confounds);
end
          
end

% Silly code to try out hmmtest
%
% Gamma = zeros(10000,6);
% T = 100 * ones(100,1);
% Tsubject = [2500 2500 2500 2500]';
% D = zeros(100,3);
% for j = 1:100
%     g = zeros(100,6);
%     t = 1; 
%     while t < 100
%         k = find(mnrnd(1,ones(1,6)/6));
%         l = 2*poissrnd(5);
%         g(t:t+l-1,k) = 1;
%         t = t+l;
%     end
%     g = g(1:100,:);
%     Gamma((1:100)+(j-1)*100,:) = g;
%     D(j,:) = mnrnd(1,ones(1,3)/3);
% end
% options = struct();
% options.Nperm = 1000;
% options.subjectlevel = 1; 
% options.grouplevel = 1; 
% tests = hmmtest (Gamma,T,Tsubject,D,options);



