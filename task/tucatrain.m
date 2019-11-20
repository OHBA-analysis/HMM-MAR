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
    options.classifier = 'logistic'; %set as default
end
classifier = options.classifier;

% Check options and put data in the right format
[X,Y,T,options,A,stats.R2_pca,npca,features] = preproc4hmm(X,Y,T,options); 
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
if isfield(options,'updateGamma') && options.updateGamma
    options.updateGamma = 1;

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

end

