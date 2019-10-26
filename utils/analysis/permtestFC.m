function [P,Psession,D] = permtestFC(data,T,options)
%
% Test the how different are the states in terms of FC
%
% INPUT
% data          observations, either a struct with X (time series) and C (classes, optional)
%                             or just a matrix containing the time series
% T             length of series
% options       structure with the training options. Must contain a field
%               options.nperm, with the number of permutations for permutation testing; 
%               and can contain a field options.threshold (between 0 and 1), which is the
%               minimum occupancy for a state to be considered active within a time series
%
% OUTPUT
% P      (K by K) matrix of p-values for the hypothesis: is the FC of every
%           pair of states different?
% Psession (no. of time series by 1) vector, indicating, for each time
%           series (i.e. for each element of T), whether there are at least two 
%           states that are significantly different in terms of their FC.
%           For a state to be considered present in a time series, it is
%           required to occupy a certain minimum percentage of the time, as
%           indicated in the field options.threshold. The values are in the
%           format of p-values, i.e. if Psession(1) = 0.0001 then it
%           indicates that in the first time series there are states with
%           statistically different FC. However, these shouldn't be
%           interpreted as p-values for the hypothesis "for this time
%           series, there is FC switching", because there's not a formal
%           statistical test for that here. 
% D         Permutation values
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

if ~isfield(options,'nperm')
   warning('options.nperm will be set to 1000') 
   options.nperm = 1000;
end
nperm = options.nperm;
options = rmfield(options,'nperm');

if ~isfield(options,'threshold')
   warning('options.threshold will be set to 0.1') 
   options.threshold = 0.1; 
end
threshold = options.threshold;
options = rmfield(options,'threshold');

if isfield(options,'order') && options.order > 0
    warning('options.order must be 0')
    options.order = 0;
end
if isfield(options,'zeromean') && options.zeromean == 0
    warning('options.zeromean must be 1')
    options.zeromean = 1; 
end
if isfield(options,'covtype') && ~strcmp(options.covtype,'full')
    warning('options.covtype must be ''full'' ')
end
options.covtype = 'full';

% is this going to be using the stochastic learning scheme? 
stochastic_learn = isfield(options,'BIGNbatch') && (options.BIGNbatch < N && options.BIGNbatch > 0);
if stochastic_learn
    error('Stochastic learning is not yet implemented within permtestFC')
end
options = checkspelling(options);

if xor(iscell(data),iscell(T)), error('X and T must be cells, either both or none of them.'); end

if iscell(T)
    T = cell2mat(T);
end
checkdatacell;
[options,data] = checkoptions(options,data,T);
K = options.K; ndim = size(data.X,2); 
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
    ndim = size(options.A,2);
    A = options.A;
else
    options.ndim = size(data.X,2);
    A = [];
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

% estimate the HMMs 
[hmm_init,~,~,vp] = hmmmar(data,T,options); 

D = zeros(nperm,K,K);

for iperm = 1:nperm
    if iperm > 1
        vp_p = surrogate_VP(vp);
    else
        vp_p = vp; 
    end
    Gamma = vpath_to_stc(vp_p,K);
    hmm = hmm_init;
    setxx;
    hmm = obsupdate(T,Gamma,hmm,data.X,XX,XXGXX);
    for k1 = 1:K-1
        [~,C1] = getFuncConn(hmm,k1);
        if ~isempty(A), C1 = A' * C1 * A; end 
        for k2 = k1+1:K
            [~,C2] = getFuncConn(hmm,k2);
            if ~isempty(A), C2 = A' * C2 * A; end 
            d = riemannian_dist(C1,C2);
            D(iperm,k1,k2) = d; 
            D(iperm,k2,k1) = d; 
        end
    end
    %if rem(iperm,10)==0, disp(['Permutation ' num2str(iperm)]); end
end

% Do the testing
P = eye(K);
for k1 = 1:K-1
   for k2 = k1+1:K
       P(k1,k2) = sum(D(1,k1,k2) <= D(:,k1,k2)) / (nperm-1);
       P(k2,k1) = P(k1,k2);
   end
end

% See if there is FC switching within each time series
Psession = [];
if nargout > 1
    Psession = zeros(N,1);
    Gamma = vpath_to_stc(vp,K);
    FO = getFractionalOccupancy (Gamma,T,options,2);
    for j = 1:N
        ind = find(FO(j,:) > threshold);
        Pj = P(ind,ind);
        Psession(j) = min(Pj(:));
    end
end


end



function vp_s = surrogate_VP(vp)
% Create a surrogate version of the specified Viterbi Path
% by randomise the labels within each visit.
K = max(vp);
prob_states = zeros(1,K);
for k = 1:K, prob_states(k) = mean(vp==k); end
% this is to have all states visited
prob_states = prob_states + 0.05; prob_states = prob_states / sum(prob_states);
change_points = find(abs(diff(vp))>0) + 1;
new_values = my_randi(length(change_points)+1,prob_states);
vp_s = zeros(size(vp));
vp_s(1:change_points(1)-1) = new_values(1);
for j = 2:length(change_points)
    vp_s(change_points(j-1):change_points(j)-1) = new_values(j);
end
vp_s(change_points(end):end) = new_values(end);
end

function r = my_randi(N,p)
% generate N random integers given the probabilities given in p
K = length(p);
rr = rand(N,1);
r = ones(N,1);
cp = cumsum(p);
for k = 1:K-1
    r = r + (rr>cp(k));
end
end
