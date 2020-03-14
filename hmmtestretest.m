function [C,hmm] = hmmtestretest(data,T,options)
%
% Test the reliability of the solutions across repetitions of the HMM
% inference. Runs are aligned using the Hungarian algorithm and compared
% for each subject/segment (each element of T). The compared metric is the
% amount of agreement between the state time courses.
% This is computed as the mean joint probabilities between the runs.
% For example, if state 1 has activation [0 0 1] in one run and [0 0.5 0.5]
% on the other, the metric would be (0*0 + 0*0.5 + 1*0.5) = 0.5 .
% Given that we sum across states, the maximum value is 1, and the minimum is 0
%
% INPUT
% data          observations, either a struct with X (time series) and C (classes, optional)
%                             or just a matrix containing the time series
% T             length of series
% options       structure with the training options - see documentation.
%               Must contain a field testretest_rep with the number of
%               repetitions of the HMM inference
%
% OUTPUT
% C      (subjects by runs by runs) matrix of state time courses
%           "agreements", for each subject, and for each pair of runs
% hmm    (K by 1) cell of hmm structures
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2019)

if iscell(T)
    if size(T,1)==1, T = T'; end
    for i = 1:length(T)
        if size(T{i},1)==1, T{i} = T{i}'; end
    end
end
N = length(T);

if isstruct(data) && isfield(data,'C')
    error('C cannot be specified within data here')
end

if ~isfield(options,'testretest_rep')
   error('options must contain a ''testretest_rep'' field') 
end
nrep = options.testretest_rep;
options = rmfield(options,'testretest_rep');

% is this going to be using the stochastic learning scheme? 
stochastic_learn = isfield(options,'BIGNbatch') && (options.BIGNbatch < N && options.BIGNbatch > 0);
if stochastic_learn
    error('Stochastic learning cannot currently be used within hmmtestretest')
end
options = checkspelling(options);

if xor(iscell(data),iscell(T)), error('X and T must be cells, either both or none of them.'); end

if iscell(T)
    T = cell2mat(T);
end
checkdatacell;
[options,data] = checkoptions(options,data,T);
K = options.K; 
if isfield(options,'orders')
    options = rmfield(options,'orders');
end
if isfield(options,'maxorder')
    options = rmfield(options,'maxorder');
end

options.dropstates = 0;
options.updateGamma = options.K>1;
options.updateP = options.updateGamma;

%%% Preprocessing
% Standardise data and control for ackward trials
data = standardisedata(data,T,options.standardise);
% Filtering
if ~isempty(options.filter)
    data = filterdata(data,T,options.Fs,options.filter); options.filter = [];
end
% Detrend data
if options.detrend
    data = detrenddata(data,T); options.detrend = 0;
end
% Leakage correction
if options.leakagecorr ~= 0
    data = leakcorr(data,T,options.leakagecorr); options.leakagecorr = 0;
end
% Hilbert envelope
if options.onpower
    data = rawsignal2power(data,T); options.onpower = 0;
end
% Leading Phase Eigenvectors
if options.leida
    data = leadingPhEigenvector(data,T); options.leida = 0;
end
% pre-embedded PCA transform
if length(options.pca_spatial) > 1 || (options.pca_spatial > 0 && options.pca_spatial ~= 1)
    if isfield(options,'As')
        data.X = bsxfun(@minus,data.X,mean(data.X));
        data.X = data.X * options.As;
    else
        [options.As,data.X] = highdim_pca(data.X,T,options.pca_spatial);
        options.pca_spatial = size(options.As,2);
    end
    options.pca_spatial = [];
end
% Embedding
if length(options.embeddedlags) > 1
    [data,T] = embeddata(data,T,options.embeddedlags); options.embeddedlags = 0;
end
% PCA transform
if length(options.pca) > 1 || (options.pca > 0 && options.pca ~= 1)
    if isfield(options,'A')
        data.X = bsxfun(@minus,data.X,mean(data.X));
        data.X = data.X * options.A;
    else
        options.A = highdim_pca(data.X,T,options.pca,0,0,0,options.varimax);
    end
else
    options.ndim = size(data.X,2);
end
% Downsampling
if options.downsample > 0
    [data,T] = downsampledata(data,T,options.downsample,options.Fs);
    options.downsample = 0;
end
if options.pcamar > 0 && ~isfield(options,'B')
    % PCA on the predictors of the MAR regression, per lag: X_t = \sum_i X_t-i * B_i * W_i + e
    options.B = pcamar_decomp(data,T,options);
    options.pcamar = 0;
end
if options.pcapred > 0 && ~isfield(options,'V')
    % PCA on the predictors of the MAR regression, together:
    % Y = X * V * W + e, where X contains all the lagged predictors
    % So, unlike B, V draws from the temporal dimension and not only spatial
    options.V = pcapred_decomp(data,T,options);
    options.pcapre = 0;
end

hmm = cell(1,nrep); Gamma = cell(1,nrep);

% estimate the HMMs 
for irun = 1:nrep
    [hmm{irun},Gamma{irun}] = hmmmar(data,T,options); 
    Gamma{irun} = padGamma(Gamma{irun},T,options);
end

% re-order the states
for irun = 2:nrep
    [~,assig, Gamma{irun}] = getGammaSimilarity (Gamma{irun}, Gamma(1:irun-1));
    hmm{irun}.state(assig) = hmm{irun}.state;
    hmm{irun}.Pi(assig) = hmm{irun}.Pi;
    hmm{irun}.Dir_alpha(assig) = hmm{irun}.Dir_alpha; 
    hmm{irun}.P(assig,assig) = hmm{irun}.P;
    hmm{irun}.Dir2d_alpha(assig,assig) = hmm{irun}.Dir2d_alpha;     
end

% compute final statistic
C = zeros(N,nrep,nrep);
for j = 1:N 
   ind = (1:T(j)) + sum(T(1:j-1));
   C(j,:,:) = eye(nrep);
   for irun = 1:nrep-1
       for irun_2 = irun+1:nrep
           for k = 1:K
               C(j,irun,irun_2) = C(j,irun,irun_2) +  sum(min(Gamma{irun}(ind,k), Gamma{irun_2}(ind,k)))  / T(j);
           end
           C(j,irun_2,irun) = C(j,irun,irun_2);
       end
   end
end

end

