function [Path,stats] = decodeBigHMM(files,T,centroids,markovTrans,prior,type,options)
% 1) Compute the viterbi paths 
% 2) Compute the entropy,avglifetime,nochanges from it
%
% INPUTS
% files: cell with strings referring to the subject files
% T: cell of vectors, where each element has the length of each trial per
% centroids: the metastates computed from trainBigHMM
% markovTrans.P and Pi: transition prob table and initial prob
% prior: prior distributions as returned by trainBigHMM
% type: 0, state time courses; 1, viterbi path
% options: HMM options for both the subject and the group runs
%
% Diego Vidaurre, OHBA, University of Oxford (2015)

N = length(files);
K = length(centroids);
if ~isfield(options,'covtype'), options.covtype = 'full'; end

X = loadfile(files{1});
hmm0 = initializeHMM(X,T{1},K,options.covtype);

AverageLifeTime = zeros(N,1);
NoChanges = zeros(N,1);
Entropy = zeros(N,1);
Path = [];

P = markovTrans.P;
Pi = markovTrans.Pi;
Dir2d_alpha = markovTrans.Dir2d_alpha;
Dir_alpha = markovTrans.Dir_alpha;

for i = 1:N
    X = loadfile(files{i});        
    if strcmp(options.covtype,'uniquefull') 
        gram = X' * X;
    elseif strcmp(options.covtype,'uniquediag')
        gram = sum(X.^2);
    else
        gram = [];
    end  
    hmm = loadhmm(hmm0,T{i},K,centroids,P{i},Pi{i},Dir2d_alpha{i},Dir_alpha{i},gram,prior);
    if type==0
        data = struct('X',X,'C',NaN(size(X,1),K));
        gamma = hsinference(data,T{i},hmm,[]);
        Path = cat(3,Path,single(gamma));
    else
        vp = hmmdecode(X,T{i},hmm,[]);
        vp = [vp(1).q_star vp(2).q_star vp(3).q_star vp(4).q_star];
        Path = cat(3,Path,int8(vp));
        gamma = zeros(numel(vp),K);
        vp = vp(:);
        for k=1:K, gamma(vp==k,k) = 1; end
    end
    slt=collect_times(gamma);
    AverageLifeTime(i) = mean(slt);
    NoChanges(i) = length(slt);
    gammasum = sum(gamma);
    gammasum = gammasum + eps;
    Entropy(i) = sum(log(gammasum) .* gammasum);
end

stats = struct('AverageLifeTime',AverageLifeTime,...
    'NoChanges',NoChanges,...
    'Entropy',Entropy);

end


function hmm = initializeHMM(X,T,K,covtype)

options_hmm = struct();
options_hmm.K = K;
options_hmm.covtype = covtype;
options_hmm.decodeGamma = 0;
options_hmm.order = 0;
options_hmm.cyc = 1;
options_hmm.zeromean = 0;
options_hmm.inittype = 'random';
options_hmm.DirichletDiag = 10;
options_hmm.initcyc = 1;
options_hmm.initrep = 1;
options_hmm.verbose = 0;
hmm = hmmmar(X,T,options_hmm);

end
