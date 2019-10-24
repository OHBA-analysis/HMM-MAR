function [tuda,Gamma,GammaInit,vpath,stats] = tucatrain(X,Y,T,options)
% Performs the Temporal Unconstrained Classification Approach (TUDA), 
% an alternative approach for decoding where we dispense with the assumption 
% that the same decoding is active at the same time point at all trials. 
% 
% INPUT
% X: Brain data, (time by regions) or (time by trials by regions)
% Y: Stimulus, (time by q); q is no. of stimulus features OR
%              (no.trials by q), meaning that each trial has a single
%              stimulus value
% T: Length of series
% options: structure with the training options - see documentation in 
%                       https://github.com/OHBA-analysis/HMM-MAR/wiki
%
% OUTPUT
% tuda: Estimated TUDA model (similar to an HMM structure)
%       It contains the fields of any HMM structure. It also contains:
%           - features: if feature selection is performed, which
%               features have been used
% Gamma: Time courses of the states (decoding models) probabilities given data
% GammaInit: Initialisation state time courses, where we do assume
%       that the same decoding is active at the same time point at all trials.
% vpath: Most likely state path of hard assignments 
% stats: structure with additional information
%   - R2_pca: explained variance of the PCA decomposition used to
%       reduce the dimensionality of the brain data (X)
%   - fe: variational free energy 
%   - R2: (training) explained variance of tuda (time by q) 
%   - R2_states: (training) explained variance of tuda per state (time by q by K) 
%   - R2_stddec: (training) explained variance of the standard 
%               (temporally constrained) decoding approach (time by q by K) 
%
% Author: Cam Higgins, OHBA, University of Oxford (2017)

stats = struct();
N = length(T); 

if ~isfield(options,'classifier')
    options.classifier='logistic'; %set as default
end
classifier=options.classifier;


% Check options and put data in the right format
[X,Y,T,options,stats.R2_pca,npca,features] = preproc4hmm(X,Y,T,options); 
parallel_trials = options.parallel_trials; 
options = rmfield(options,'parallel_trials');
if isfield(options,'add_noise'), options = rmfield(options,'add_noise'); end
p = size(X,2); q = size(Y,2);
 
% init HMM, only if trials are temporally related:
if options.sequential
    GammaInit = zeros(T(1),options.K);
    incs=ceil(options.K*(1:T(1))./T(1));
    GammaInit(sub2ind(size(GammaInit), 1:T(1), incs)) = 1;
    options.Gamma = permute(repmat(GammaInit,[1 1 N]),[1 3 2]);
    options.Gamma = reshape(options.Gamma,[length(T)*size(GammaInit,1) options.K]);
elseif parallel_trials && ~isfield(options,'Gamma')
    GammaInit = cluster_decoding(X,Y,T,options.K,'regression','',...
        options.Pstructure,options.Pistructure);
    options.Gamma = permute(repmat(GammaInit,[1 1 N]),[1 3 2]);
    options.Gamma = reshape(options.Gamma,[length(T)*size(GammaInit,1) options.K]);
elseif ~parallel_trials
    GammaInit = [];
else 
    GammaInit = options.Gamma;
end
options=rmfield(options,'sequential');

% if cyc==0 there is just init and no HMM training 
if isfield(options,'cyc') && options.cyc == 0 
   if ~parallel_trials
      error('Nothing to do, specify options.cyc > 0') 
   end
   tuda = []; vpath = [];  
   Gamma = options.Gamma;
   stats.R2_stddec = R2_standard_dec(X,Y,T);
   return
end


% Put X and Y together
Ttmp = T;
T = T + 1;
Z = zeros(sum(T),q+p,'single');
for n=1:N
    t1 = (1:T(n)) + sum(T(1:n-1));
    t2 = (1:Ttmp(n)) + sum(Ttmp(1:n-1));
    if strcmp(classifier,'logistic')
        Z(t1(1:end-1),1:p) = X(t2,:);
        Z(t1(2:end),(p+1):end) = Y(t2,:);
    elseif strcmp(classifier,'LDA')
        Z(t1(2:end),1:p) = X(t2,:);
        Z(t1(1:end-1),(p+1):end) = Y(t2,:);
    end
end 

% Run TUDA inference
options.S = -ones(p+q);
if strcmp(classifier,'logistic')
    options.S(1:p,p+1:end) = 1;
elseif strcmp(classifier,'LDA')
    options.S(p+1:end,1:p) = 1;
end

%switch off parallel as not implemented:
options.useParallel=0;
options.decodeGamma=0;

% 1. Estimate Obs Model parameters given Gamma, unless told not to:
options_run1=options;
if isfield(options,'updateObs') 
    options_run1.updateObs=1
end 
options_run1.updateGamma=0;
options_run1.decodeGamma=0;

[tuda,Gamma,~,vpath] = hmmmar(Z,T,options_run1);
options = rmfield(options,'Gamma');

% 2. Update state time courses only, leaving fixed obs model params:
if ~isfield(options,'updateGamma')
    options.updateGamma = 1;
end
options.updateObs = 1; % 
%options.Gamma = Gamma;
options.hmm = tuda; 
if ~isfield(options,'cyc')
    options.cyc=4;
end
tudamonitoring = options.tudamonitoring;
if isfield(options,'behaviour')
    behaviour = options.behaviour;
else 
    behaviour = [];
end
options.tudamonitoring = 0;
options.behaviour = [];
options.verbose = 1;
warning off
[tuda,Gamma,~,~,~,~, stats.fe] = hmmmar(Z,T,options); 
warning on

tuda.features = features;
options.tudamonitoring = tudamonitoring;
options.behaviour = behaviour;
%options.verbose = verbose;

tuda.train.pca = npca;

end

function Gamma = cluster_decoding(X,Y,T,K,cluster_method,...
    cluster_measure,Pstructure,Pistructure)
% clustering of the time-point-by-time-point regressions, which is
% temporally constrained unlike TUDA
% INPUT
% X,Y,T are as usual
% K is the number of states 
% cluster_method is 'regression', 'kmeans', or 'hierarchical' 
% cluster_measure is 'error', 'response' or 'beta' 
% Pstructure and Pistructure are constraints in the transitions 
%   if these are specified, cluster_method will be set to 'greedy'
% OUTPUT
% Gamma: (trial time by K), containing the cluster assignments 

if nargin<5, cluster_method = 'regression'; end
if nargin>5 && ~isempty(cluster_measure) && strcmp(cluster_method,'regression')
   warning('cluster_measure is not used when cluster_method is regression') 
end
if nargin<6, cluster_measure = 'error'; end
if nargin<7, Pstructure = true(K,1); end
if nargin<8, Pistructure = true(K); end
if ~strcmp(cluster_measure,'beta') && strcmp(cluster_method,'kmeans')
    error('If kmeans is used, cluster_measure must be beta')
end
N = length(T); p = size(X,2); q = size(Y,2); ttrial = T(1);
X = reshape(X,[ttrial N p]);
Y = reshape(Y,[ttrial N q]);
if strcmp(cluster_method,'regression')
    max_cyc = 100;
    % start with no constraints
    Gamma = cluster_decoding(reshape(X,[ttrial*N p]),reshape(Y,[ttrial*N q]),...
        T,K,'hierarchical','error');
    assig = zeros(ttrial,1);
    for t=1:ttrial, assig(t) = find(Gamma(t,:)==1); end
    j1 = assig(1);
    if ~Pistructure(j1) % is it consistent with constraint?
        j = find(Pistructure,1); 
        Gamma_j = Gamma(:,j);
        Gamma(:,j) = Gamma(:,j1);
        Gamma(:,j1) = Gamma_j;
        for t=1:ttrial, assig(t) = find(Gamma(t,:)==1); end
    end
    assig_pr = assig;
    beta = zeros(p,q,K);
    err = zeros(ttrial,K);
    for cyc = 1:max_cyc
        % M
        for k=1:K
           ind = assig==k;
           Xstar = reshape(X(ind,:,:),[sum(ind)*N p]);
           Ystar = reshape(Y(ind,:,:),[sum(ind)*N q]);
           beta(:,:,k) = (Xstar' * Xstar) \ (Xstar' * Ystar);
        end
        % E
        for k=1:K
           Yhat = reshape(X,[ttrial*N p]) * beta(:,:,k); 
           e = sum((reshape(Y,[ttrial*N q]) - Yhat).^2,2);
           e = reshape(e,[ttrial N]); 
           err(:,k) = sqrt(sum(e,2));
        end
        err(1,~Pistructure) = Inf; 
        [~,assig(1)] = min(err(1,:));
        for t = 2:ttrial
           err(t,~Pstructure(assig(t-1),:)) = Inf;
           [~,assig(t)] = min(err(t,:)); 
        end
        % terminate? 
        if all(assig_pr==assig), break; end
        assig_pr = assig;
    end
    for t = 1:ttrial
        Gamma(t,:) = 0;
        Gamma(t,assig(t)) = 1;
    end
else
    beta = zeros(p,q,ttrial);
    for t = 1:ttrial
        Xt = permute(X(t,:,:),[2 3 1]);
        Yt = permute(Y(t,:,:),[2 3 1]);
        beta(:,:,t) = (Xt' * Xt) \ (Xt' * Yt);
    end
    if strcmp(cluster_measure,'response')
        dist = zeros(ttrial*(ttrial-1)/2,1);
        Xstar = reshape(X,[ttrial*N p]);
        c = 1; 
        for t2 = 1:ttrial-1
            d2 = Xstar * beta(:,:,t2);
            for t1 = t2+1:ttrial
                d1 = Xstar * beta(:,:,t1);
                dist(c) = sqrt(sum(sum((d1 - d2).^2))); c = c + 1; 
            end
        end
    elseif strcmp(cluster_measure,'error')
        dist = zeros(ttrial*(ttrial-1)/2,1);
        c = 1;
        for t2 = 1:ttrial-1
            Xt2 = permute(X(t2,:,:),[2 3 1]);
            Yt2 = permute(Y(t2,:,:),[2 3 1]);
            for t1 = t2+1:ttrial
                Xt1 = permute(X(t1,:,:),[2 3 1]);
                Yt1 = permute(Y(t1,:,:),[2 3 1]);
                error1 = sqrt(sum(sum((Xt1 * beta(:,:,t2) - Yt1).^2)));
                error2 = sqrt(sum(sum((Xt2 * beta(:,:,t1) - Yt2).^2)));
                dist(c) = error1 + error2; c = c + 1; 
            end
        end
    elseif strcmp(cluster_measure,'beta')
        beta = permute(beta,[3 1 2]);
        beta = reshape(beta,[ttrial p*q]);
        if strcmp(cluster_method,'hierarchical')
            dist = pdist(beta);
        end
    end
    if strcmp(cluster_method,'kmeans')
        assig = kmeans(beta,K);
    else % hierarchical
        if iseuclidean(dist')
            link = linkage(dist','ward');
        else
            link = linkage(dist');
        end
        assig = cluster(link,'MaxClust',K);
    end
end
Gamma = zeros(ttrial, K);
for k = 1:K
    Gamma(assig==k,k) = 1;
end
end


function mGamma = meanGamma(Gamma,T)
% Get the trial-averaged state time courses
N = length(T); ttrial = sum(T)/N;
K = size(Gamma,2);
mGamma = squeeze(mean(reshape(Gamma,[ttrial N K]),2));
end


function t = iseuclidean(D)
% D is (1 by pairs)
m = size(D,2);
% make sure it's a valid dissimilarity matrix
n = ceil(sqrt(2*m)); % (1+sqrt(1+8*m))/2, but works for large m
if n*(n-1)/2 == m && all(D >= 0)
    D = squareform(D);
else
    warning(message('stats:iseuclidean:NotDistanceMatrix'))
    t = false;
    return
end
P = eye(n) - repmat(1/n,n,n);
B = P * (-.5 * D .* D) * P;
g = eig((B+B')./2); % guard against spurious complex e-vals from roundoff
t = all(-eps(class(g))^(3/4) * max(abs(g)) <= g); 
% all non-negative eigenvals (within roundoff)?
end
