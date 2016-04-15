function stats = getshmmstats(Path,T)
% 1) Compute the stata time courses or viterbi paths 
% 2) Compute the entropy,avglifetime,nochanges from it
%
% INPUTS
% Xin: cell with strings referring to the subject files
% T: cell of vectors, where each element has the length of each trial per
% metahmm: the metastates computed from trainBigHMM
% markovTrans.P and Pi: transition prob table and initial prob
% type: 0, state time courses; 1, viterbi path
%
% Diego Vidaurre, OHBA, University of Oxford (2015)

for i = 1:length(T)
    if size(T{i},1)==1, T{i} = T{i}'; end
end

N = length(T);
TT = []; for i=1:N, TT = [TT; T{i}]; end
tacc = 0; 

AverageLifeTime = zeros(N,1);
NoChanges = zeros(N,1);
Entropy = zeros(N,1);

for i = 1:N
    t = (1:(sum(T{i})-length(T{i})*metahmm.train.order)) + tacc;
    tacc = tacc + length(t);
    gamma = Path(t,:);
    if nargout==2
        slt=collect_times(gamma,TT);
        AverageLifeTime(i) = mean(slt);
        NoChanges(i) = length(slt);
        gammasum = sum(gamma);
        gammasum = gammasum + eps;
        Entropy(i) = sum(log(gammasum) .* gammasum);
    end
end

stats = struct('AverageLifeTime',AverageLifeTime,...
    'NoChanges',NoChanges,...
    'Entropy',Entropy);

end