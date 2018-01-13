function tests = hmmtest (Gamma,T,Tsubject,Y,options)
% 
% Test fractional occupancy and switching rates against a 
% (no. trials by no. conditions) design matrix Y using permutation testing.
% Fractional occupancy is tested per state; the switching rate
% is tested also for all states simultaneously. 
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
% Y: (no. trials by no. conditions) design matrix 
% options: testing related options
%   .Nperm: number of permutations (default 1000)
%   .subjectlevel: run subject-level tests? 
%           (default 1 if all subjects have at least 100 trials)
%   .grouplevel: run group-level tests? 
%           (default 1, if there are at least 2 subjects)
%   .confounds: (no. trials by q) matrix of confounds that are to be
%           regressed out before doing the testing (default none)
% 
% OUTPUTS:
%
% tests: a struct with fields
%   .subjectlevel: subject-level tests, with fields containing  
%           p-values for the different measures:
%       .p_fractional_occupancy: (no. states by no. subjects)   
%       .p_switching_rate: (1 by no. subjects) one p-value 
%               for the global rate of switching
%   .grouplevel: group-level tests, with fields
%       .p_fractional_occupancy: (no. states by 1)  
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
if size(Tsubject,1)==1, Tsubject = Tsubject'; end

if isempty(Tsubject), Tsubject = sum(T); end  % One subject
if sum(T) ~= sum(Tsubject), error('sum(T) must equal to sum(Tsubject)'); end
N = length(T); Nsubj = length(Tsubject);
order =  (sum(T) - size(Gamma,1)) / N;
K = size(Gamma,2); % no. of states
%threshold_visit = 3; % less that this no. of time point, a state visit is spurious
%threshold_Gamma = (2/3); % threshold over which a state is deemed to be active

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


% Adjust dimensions of T and Tsubject
if order>0 
    Tsubject = Tsubject - Ntrials_per_subject * order; 
    T = T - order;
end

tests = struct();
tests.subjectlevel = struct();
tests.grouplevel = struct();

% Testing at the subject level
if subjectlevel
    tests.subjectlevel.p_fractional_occupancy = zeros(K,Nsubj);
    %tests.subjectlevel.p_life_times = zeros(K,Nsubj);
    %tests.subjectlevel.p_interval_times = zeros(K,Nsubj); 
    tests.subjectlevel.p_switching_rate = zeros(1,Nsubj);  
    for j = 1:Nsubj
        t = (1:Tsubject(j)) + sum(Tsubject(1:j-1));
        g = Gamma(t,:);
        jj = (1:Ntrials_per_subject(j)) + sum(Ntrials_per_subject(1:j-1));
        Yj = Y(jj,:);
        fo = getFractionalOccupancy(g,TperSubj{j});
        %lt = getStateLifeTimes(g,TperSubj{j},threshold_visit,threshold_Gamma);
        %it = getStateIntervalTimes(g,TperSubj{j},threshold_visit,threshold_Gamma);
        sr = getSwitchingRate(g,TperSubj{j});
        tests.subjectlevel.p_fractional_occupancy(:,j) = permtest(fo,Yj,Nperm,[],confounds);
        %tests.subjectlevel.p_life_times(:,j) = permtest(lt,Yj,Nperm,[],confounds);
        %tests.subjectlevel.p_interval_times(:,j) = permtest(it,Yj,Nperm,[],confounds);
        tests.subjectlevel.p_switching_rate(j) = permtest(sr,Yj,Nperm,[],confounds);
    end
end
   
% Testing at the group level
if grouplevel
    if (length(T)==length(Tsubject)) 
        Ntrials_per_subject = [];
    end
    fo = getFractionalOccupancy(Gamma,T);
    %lt = getStateLifeTimes(Gamma,T,threshold_visit,threshold_Gamma);
    %it = getStateIntervalTimes(Gamma,T,threshold_visit,threshold_Gamma);
    sr = getSwitchingRate(Gamma,T);
    tests.grouplevel.p_fractional_occupancy = permtest(fo,Y,Nperm,Ntrials_per_subject,confounds);
    %tests.grouplevel.p_life_times(:,j) = permtest(lt,Y,Nperm,Ntrials_per_subject,confounds);
    %tests.grouplevel.p_interval_times(:,j) = permtest(it,Y,Nperm,Ntrials_per_subject,confounds);
    tests.grouplevel.p_switching_rate = permtest(sr,Y,Nperm,Ntrials_per_subject,confounds);
end
          
end


function pval = permtest(X,D,Nperm,grouping,confounds)
% permutation testing routine (through regression)
% X: data
% D: design matrix
% Nperm: no. of permutations
% grouping: the first grouping(1) rows belong to one group, 
%    the next grouping(2) rows belong to the second group, 
%    and so on and so on
% Diego Vidaurre

[N,p] = size(X);
if nargin<4, grouping = []; end
if (nargin>4) && ~isempty(confounds)
    confounds = confounds - repmat(mean(confounds),N,1);
    X = X - confounds * pinv(confounds) * X;
end

X = X - repmat(mean(X),size(X,1),1);  
grotperms = zeros(Nperm,p);
proj = (D' * D + 0.001 * eye(size(D,2))) \ D';  

for perm=1:Nperm
    if perm==1
       Xin = X;
    else
        if ~isempty(grouping)
            Xin = zeros(size(X));
            for gr = 1:length(grouping)
                jj = (1:grouping(gr)) + sum(grouping(1:gr-1));
                r = randperm(grouping(gr));
                Xin(jj,:) = X(jj(r),:);
            end
        else
            r = randperm(N);
            Xin = X(r,:);
        end
    end
    beta = proj * Xin;
    grotperms(perm,:) = sqrt(sum((D * beta - Xin).^2)); 
end

pval = zeros(p,1);
for j = 1:p
    pval(j) = sum(grotperms(:,j)<=grotperms(1,j)) / (Nperm+1);
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



