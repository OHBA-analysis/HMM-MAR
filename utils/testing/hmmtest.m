function tests = hmmtest (Gamma,T,Tsubject,Y,options,hmm,Xi)
% Statistical testing to differenciate between groups according to how much
% time they spend on the different states, or how quickly they switch between states,
% or how their transition probability matrices (TPM) differ. 
% Tests are done at the subject or at the group level.
% There are two types of group level tests: 
% (i) when Y (the behavioural variable, see below) has one element per
% subject; then permutations are done across subjects. That means we are
% testing here subject differences
% (ii) when Y has one element per trial/session; then permutations are run within
% subject even though the p-value is obtained at the group level. That
% means that we are testing for between-trial/session differences. 
% Tests are *not* corrected for multiple comparisons. 
% 
%For statistical testing on how the probability of the states is modulated
% by a stimulus, time point by time point, use hmmtest_epoched.  
%
% More specifically, it tests fractional occupancy, switching rates 
% and transition probability matrics against a 
% (no. trials/subjects by no. conditions) design matrix Y using permutation testing.
% Fractional occupancy is tested per state, and also aggregated across states
%   using the NPC algorithm; see Winkler et al., 2016 NI; Vidaurre et al. 2018 HBM
%   (in this case correction for multiple comparisons across states is not necessary).
% Switching rate is tested also for all states simultaneously. 
% Transition probability matrices are tested element by element after
%   computing the subject- or trial-specific transition matrices in a
%   dual-estimation process
% Tests can be paired (options.paired=1) or unpaired (options.paired=0, by default). 
% If paired options.pairs is used to mark the pairs. For example, if there
%  are 4 subjects, then options.pairs = [1 1 2 2 3 3 4 4] if the
%  paired subjects are consecutive; ie the length of options.pairs
%  is that of the number of subjects.
%
% INPUTS:
%
% Gamma: state time courses, as returned by hmmmar
% T: length of time series (no. trials/sessions by 1) in number of time points
% Tsubject: length of time series per subject (no. subjects by 1); 
%       if empty it will be interpreted that there is only one subject
% Y: (no. trials by no. conditions) design matrix if testing is done at the subject level
%   or (no. subjects by no. conditions) if testing is  done at the group level.
% options: testing related options
%   .Nperm: number of permutations (default 1000)
%   .subjectlevel: run subject-level tests? if 0, group testing will be done
%           (if unspecified it will be guessed based on Y)
%   .confounds: (no. trials by q) matrix of confounds that are to be
%           regressed out before doing the testing (default none)
%   .paired: paired tests? default: false
%   .pairs: if so, a vector indicating the pairs (eg [ 1 1 2 2 ... ])
%   .testP: test on the transition probability matrices? 
% hmm: hmm structure
% Xi: joint probability of past and future states conditioned on data
%      (optional and only used if options.testP==true)
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
if isempty(Tsubject), Tsubject = sum(T); end  % One subject
if iscell(Tsubject)
    if size(Tsubject,1)==1, Tsubject = Tsubject'; end
    Tsubject = cell2mat(Tsubject);
end
if size(T,1)==1, T = T'; end
if size(Tsubject,1)==1, Tsubject = Tsubject'; end
if sum(T) ~= sum(Tsubject), error('sum(T) must equal to sum(Tsubject)'); end
N = length(T); Nsubj = length(Tsubject);
K = size(Gamma,2); % no. of states
if nargin<7, Xi = []; end
if nargin<6 || isempty(hmm), hmm = struct(); hmm.train = struct(); end

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
    TperSubj = []; Ntrials_per_subject = N;
end

% Check options
if ~isfield(options,'Nperm'), Nperm = 1000; 
else, Nperm = options.Nperm;
end
if isfield(options,'subjectlevel'), subjectlevel = options.subjectlevel;
else, subjectlevel = ~(size(Y,1) == Nsubj);
end
if subjectlevel && min(Ntrials_per_subject)<10
    warning('Not enough trials per subject in order to get subject-specific p-values ')
    subjectlevel = 0;
end
grouplevel = ~subjectlevel;
if isfield(options,'paired'), paired = options.paired;
else, paired = false; 
end
if paired 
    if isfield(options,'pairs'), pairs = options.pairs; 
    else, pairs = repmat(1:Nsubj/2,2,1); pairs = pairs(:); 
    end
else
    pairs = [];
end
if paired && subjectlevel
   error('paired tests only implemented at the group level') 
end
if isfield(options,'testP'), testP = options.testP;
else, testP = false; 
end
if ~isfield(options,'confounds'), confounds = [];
else, confounds = options.confounds;
end
if ~isfield(hmm.train,'downsample') || hmm.train.downsample==0 || ...
        ~isfield(hmm.train,'Fs')
    r = 1; 
else
    r = hmm.train.downsample/hmm.train.Fs;
end

% Adjust dimensions of T and Tsubject
if isfield(hmm.train,'order') && hmm.train.order > 0
    Tsubject = ceil(r * Tsubject);
    Tsubject = Tsubject - Ntrials_per_subject * hmm.train.order; 
    T = ceil(r * T);
    T = T - hmm.train.order;
    if ~isempty(TperSubj)
        for j = 1:length(TperSubj)
            TperSubj{j} = ceil(r * TperSubj{j});
            TperSubj{j} = TperSubj{j} - hmm.train.order;
        end
    end
elseif isfield(hmm.train,'embeddedlags') && length(hmm.train.embeddedlags) > 1
    d1 = -min(0,hmm.train.embeddedlags(1));
    d2 = max(0,hmm.train.embeddedlags(end));
    Tsubject = Tsubject - Ntrials_per_subject * (d1+d2); 
    Tsubject = ceil(r * Tsubject);
    T = T - (d1+d2); 
    T = ceil(r * T);
    if ~isempty(TperSubj)
        for j = 1:length(TperSubj)
            TperSubj{j} = TperSubj{j} - hmm.train.order;
            TperSubj{j} = ceil(r * TperSubj{j});
        end
    end
end

tests = struct();

% Testing at the subject level
if subjectlevel
    tests.subjectlevel = struct();
    if length(Y)~=N
        error(['Y must have as many rows as trials, in this case ' num2str(N) ])
    end
    tests.subjectlevel.p_fractional_occupancy = zeros(K,Nsubj);
    tests.subjectlevel.p_aggr_fractional_occupancy = zeros(1,Nsubj);
    tests.subjectlevel.p_switching_rate = zeros(1,Nsubj);  
    if testP
        tests.subjectlevel.p_transProbMat = zeros(K,K,Nsubj);  
    end
    for j = 1:Nsubj
        t = (1:Tsubject(j)) + sum(Tsubject(1:j-1));
        g = Gamma(t,:);
        jj = (1:Ntrials_per_subject(j)) + sum(Ntrials_per_subject(1:j-1));
        Yj = Y(jj,:);
        fo = getFractionalOccupancy(g,TperSubj{j});
        sr = getSwitchingRate(g,TperSubj{j});
        tests.subjectlevel.p_fractional_occupancy(:,j) = permtest_aux(fo,Yj,Nperm,confounds);
        tests.subjectlevel.p_aggr_fractional_occupancy(j) = permtest_aux_NPC(fo,Yj,Nperm,confounds);
        tests.subjectlevel.p_switching_rate(j) = permtest_aux(sr,Yj,Nperm,confounds);
        if testP
            T2 = TperSubj{j}; Ntrial = length(TperSubj{j});
            P = zeros(K,K,Ntrial);
            for j2 = 1:Ntrial
                if isempty(Xi)
                    t2_1 = (1:T2(j)-1) + sum(T2(1:j-1)); t2_2 = (2:T2(j)) + sum(T2(1:j-1));
                    xi = g(t2_1,:)' * g(t2_2,:); xi = xi / sum(xi(:)); % approximation
                else
                    t2 = (1:T2(j)-1) + sum(T2(1:j-1)) - (j-1);
                    xi = Xi(t2,:,:);
                end
                P(:,:,j2) = xi;
            end
            P = reshape(P,K*K,Ntrial)';
            pv = permtest_aux(P,Yj,Nperm,confounds);
            tests.subjectlevel.p_transProbMat(:,:,j) = reshape(pv,K,K);
        end
    end
end
   
% Testing at the group level
if grouplevel
    tests.grouplevel = struct();
    if size(Y,1)==Nsubj % alternative hypothesis: are subjects different with respect to Y?
        fo1 = getFractionalOccupancy(Gamma,T);
        sr1 = getSwitchingRate(Gamma,T);
        fo = zeros(Nsubj,K);
        sr = zeros(Nsubj,1);
        for j = 1:Nsubj
            jj = (1:Ntrials_per_subject(j)) + sum(Ntrials_per_subject(1:j-1));
            if Ntrials_per_subject(j) == 1
                fo(j,:) = fo1(jj,:); sr(j) = sr1(jj);
            else
                fo(j,:) = mean(fo1(jj,:)); sr(j) = mean(sr1(jj,:));
            end
        end
        tests.grouplevel.p_fractional_occupancy = permtest_aux(fo,Y,Nperm,confounds,pairs);
        tests.grouplevel.p_aggr_fractional_occupancy = permtest_aux_NPC(fo,Y,Nperm,confounds,pairs);
        tests.grouplevel.p_switching_rate = permtest_aux(sr,Y,Nperm,confounds,pairs);
        if testP
            P1 = zeros(K,K,N);
            for j = 1:N
                if isempty(Xi)
                    t2_1 = (1:T(j)-1) + sum(T(1:j-1)); t2_2 = (2:T(j)) + sum(T(1:j-1));
                    xi = Gamma(t2_1,:)' * Gamma(t2_2,:); xi = xi / sum(xi(:)); % approximation
                else
                    t2 = (1:T(j)-1) + sum(T(1:j-1)) - (j-1);
                    xi = Xi(t2,:,:);
                end
                P1(:,:,j) = xi;
            end
            P = zeros(K,K,Nsubj);
            for j = 1:Nsubj
                jj = (1:Ntrials_per_subject(j)) + sum(Ntrials_per_subject(1:j-1));
                if Ntrials_per_subject(j) == 1
                    P(:,:,j) = P1(:,:,jj);
                else
                    P(:,:,j) = mean(P1(:,:,jj),3);
                end
            end
            tests.grouplevel.p_transProbMat = zeros(K);
            P = reshape(P,K*K,Nsubj)';
            pv = permtest_aux(P,Y,Nperm,confounds,pairs);
            tests.grouplevel.p_transProbMat = reshape(pv,K,K);
        end
    elseif size(Y,1)==N  % alternative hypothesis: are there trial differences with respect to Y?
        if paired
           error('Pair t-test can only be used when there is one row in Y per subject') 
        end
        index_subjects = zeros(N,1);
        n_acc = 0;
        for n = 1:Nsubj
            ind = (1:Ntrials_per_subject(n)) + n_acc; 
            index_subjects(ind) = n;
            n_acc = n_acc + Ntrials_per_subject(n);
        end
        fo = getFractionalOccupancy(Gamma,T);
        sr = getSwitchingRate(Gamma,T);   
        tests.grouplevel.p_fractional_occupancy = permtest_aux(fo,Y,Nperm,confounds,pairs,index_subjects);
        tests.grouplevel.p_aggr_fractional_occupancy = permtest_aux_NPC(fo,Y,Nperm,confounds,pairs,index_subjects);
        tests.grouplevel.p_switching_rate = permtest_aux(sr,Y,Nperm,confounds,pairs,index_subjects);
        if testP
            warning('Test on state probability matrix only implemented when testing between-subject differences')
        end
    else
        
        error('Y must have either as many rows as subjects or as many as sessions' )
    end

end
          
end

% Silly code to try out hmmtest
%
% ttrial = 100; N = 200; K = 3; 
% Gamma = zeros(ttrial*N,K);
% T = ttrial * ones(N,1);
% Tsubject = 2500 * ones(8,1);
% D = zeros(N,3);
% for j = 1:N
%     g = zeros(ttrial,3);
%     t = 1; 
%     while t < ttrial
%         k = find(mnrnd(1,ones(1,K)/K));
%         l = 2*poissrnd(5);
%         g(t:t+l-1,k) = 1;
%         t = t+l;
%     end
%     g = g(1:ttrial,:);
%     Gamma((1:ttrial)+(j-1)*ttrial,:) = g;
%     D(j,:) = mnrnd(1,ones(1,K)/K);
% end
% options = struct();
% options.Nperm = 1000;
% options.subjectlevel = 1; 
% options.paired = 0; 
% options.testP = 1;
% tests = hmmtest (Gamma,T,Tsubject,D,options);
% 
% ttrial = 100; N = 200; K = 3; 
% Gamma = zeros(ttrial*N,K);
% T = ttrial * ones(N,1); Nsubj = 100;
% Tsubject = 200 * ones(Nsubj,1);
% for j = 1:N
%     g = zeros(ttrial,3);
%     t = 1; 
%     while t < ttrial
%         k = find(mnrnd(1,ones(1,K)/K));
%         l = 2*poissrnd(5);
%         g(t:t+l-1,k) = 1;
%         t = t+l;
%     end
%     g = g(1:ttrial,:);
%     Gamma((1:ttrial)+(j-1)*ttrial,:) = g;
%     D(j,:) = mnrnd(1,ones(1,K)/K);
% end
% D = zeros(Nsubj,3);
% for j = 1:Nsubj
%     D(j,:) = mnrnd(1,ones(1,3)/3);
% end
% options = struct();
% options.Nperm = 1000;
% options.subjectlevel = 0; 
% options.paired = 0; 
% options.testP = 1;
% tests2 = hmmtest (Gamma,T,Tsubject,D,options);
% 
% options.paired = 1; 
% options.testP = 1;
% tests3 = hmmtest (Gamma,T,Tsubject,D,options);
% 
