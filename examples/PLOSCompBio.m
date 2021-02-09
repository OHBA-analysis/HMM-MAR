%% Simulations 1. Section "Loss of sensitivity"

% The conclusion of these simulations is:
% If the true dimensionality of the data is equal or lower than the chosen
% PCA dimension, then HMM-PCA and the HMM-Gaussian on PCA will work similarly, but
% the HMM-Gaussian on PCA could work a bit better because it has fewer parameters.
% However, if the true dimensionality of the data is higher than the chosen
% PCA dimension, which is surely always the case in reality, then HMM-PCA will
% work better because it has the opportunity to see the original data

addpath(genpath('../HMM-MAR'))

% gen_model=1 is fully synthetic, with a latent, lower-dimensional Gaussian distribution 
%   projected to a higher-dimensional space
% gen_model=2 is partly synthetic, where the states are made by manipulating 
%   HCP-related covariance matrices. 
gen_model = 2; 
% high_dim = 1 refers to the fact that the real dimensionality of the data is higher 
% than the chosen PCA dimension (not that it's truly high-dimensional)
high_dim = 1;

% ratio of probability of staying in the same state vs changing states
stickyness_par = 25;

nrep = 50; % number of repetitions of the simulation

% variables containing how accurate the estimation recovers the true 
% state time courses
beta_mixpca = zeros(nrep,1);
beta_pca = zeros(nrep,1);
beta_std = zeros(nrep,1);

if high_dim
    filename = ['out/sims_model' num2str(gen_model) '_HD.mat'];
else
    filename = ['out/sims_model' num2str(gen_model) '_LD.mat'];
end

ttrial = 1000; % length of the trial
ntrial = 10; % no. of trials
N = ttrial * ntrial;  % no. of time points
ndim = 10; % no. of channels


for r = 1:nrep
    
    switch gen_model
        
        case 1
            
            noise_factor = 0.001;
            group_factor = 0.5; % how stronger is the group level than the state level
            if high_dim
                n_latent_dim = 3;
            else
                n_latent_dim = 2;
            end
            e = [2 1.5 1.0];
            
            Y = randn(N,n_latent_dim);
            
            A = randn(n_latent_dim,ndim) .* repmat(e(1:n_latent_dim)',1,ndim);
            A1 = randn(n_latent_dim,ndim) .* repmat(e(1:n_latent_dim)',1,ndim);
            A2 = randn(n_latent_dim,ndim) .* repmat(e(1:n_latent_dim)',1,ndim);
            
            X1 = group_factor * Y * A; % group level things
            X1 = X1 + Y * A1; % subject level
            X1 = X1 + noise_factor * randn(N,ndim); % noise
            X2 = group_factor * Y * A; % group level things
            X2 = X2 + Y * A2; % subject level
            X2 = X2 + noise_factor * randn(N,ndim); % noise
            
        case 2
            
            load('YEO100_C.mat') % a real 100x100 covariance matrix from fMRI data
            
            N = ttrial * ntrial;
            noise_factor = 0.001;
            
            [V,D] = eig(C);
            e1 = 1; % shared eigenvectors
            if high_dim
                e2 = 2; % state-specific eigenvectors
            else
                e2 = 1;
            end
            m = mean(V(:,end-e2:end-e1)); s = std(V(:,end-e2:end-e1));
            
            V1 = V; V1(:,1:end-e1) = 0;
            V2 = V1;
            for j = (size(V1,2)-e2):(size(V1,2)-e1)
                v = V(:,j);
                V1(:,j) = v(randperm(length(v)));
                V2(:,j) = v(randperm(length(v)));
            end
            V1(:,end-e2:end-e1) = repmat(s,ndim,1) .* randn(ndim,e2) + repmat(m,ndim,1);
            V2(:,end-e2:end-e1) = repmat(s,ndim,1) .* randn(ndim,e2) + repmat(m,ndim,1);
            C1 = V1 * D * V1';
            C2 = V2 * D * V2';
            
            X1 = noise_factor * randn(N,ndim); % noise
            X1 = X1 + mvnrnd(zeros(N,ndim),C1);
            X2 = noise_factor * randn(N,ndim); % noise
            X2 = X2 + mvnrnd(zeros(N,ndim),C2);
            
    end
    
    T = ttrial * ones(ntrial,1);
    
    % create transition probability matrix
    P = ones(2); P(eye(2)==1) = stickyness_par;
    for k = 1:2, P(k,:) = P(k,:) / sum(P(k,:)); end
    Pi = (1/2) * ones(1,2);
    Gamma_sim = simgamma(T,P,Pi);
    
    % sample the data using Markov Monte Carlo
    X_sim = zeros(N,ndim); done = false(N,1);
    for n = 1:ntrial
        tend = n * ttrial;
        t = 1 + (n-1) * ttrial;
        while t < tend
            this_state = find(Gamma_sim(t,:)==1);
            other_state = mod(this_state,2)+1;
            deltat = find(Gamma_sim(t+1:tend,other_state),1);
            if isempty(deltat), deltat = tend - t + 1; end
            if this_state == 1
                X_sim(t:t+deltat-1,:) = X1(t:t+deltat-1,:);
            else
                X_sim(t:t+deltat-1,:) = X2(t:t+deltat-1,:);
            end
            done(t:t+deltat-1) = true;
            t = t + deltat;
        end
    end
    
    K = 2; % number of states in the HMM (set to the true number)
    
    % standard HMM-Gaussian on PCA space
    options = struct();
    options.K = K;
    options.zeromean = 1;
    options.pca = 2;
    options.cyc = 250;
    options.verbose = 0;
    [hmm_std,Gamma_std] = hmmmar(X_sim,T,options);
    beta_std(r) = getGammaSimilarity (Gamma_std, Gamma_sim);
    
    % HMM-PCA
    options = struct();
    options.K = K;
    options.zeromean = 1;
    options.lowrank = 2;
    options.verbose = 0;
    [hmm_pca,Gamma_pca] = hmmmar(X_sim,T,options);
    beta_pca(r) = getGammaSimilarity (Gamma_pca, Gamma_sim);
    
    % Mixture of PCAs
    options = struct();
    options.K = K;
    options.zeromean = 1;
    options.lowrank = 2;
    options.id_mixture = 1;
    options.verbose = 0;
    [hmm_mixpca,Gamma_mixpca] = hmmmar(X_sim,T,options);
    beta_mixpca(r) = getGammaSimilarity (Gamma_mixpca, Gamma_sim);
        
end

%% Simulations 2. Section "﻿Bias towards low-order PCA components"

addpath(genpath('../HMM-MAR'))

datadir = '/Users/admin/Work/data/HCP/TimeSeries/';
% load some fMRI data (HCP in Schaffer parcellation, 100 regions)
load([datadir 'sub1_100Yeo.mat']); X = D; clear D

% Fig 2 panel A

ndim = 4; regions = randperm(size(X,2),ndim); % get 4 random regions
X = X(1:1200,regions); X = zscore(X); % get a chunk of data

T = size(X,1);
D = diag(randn(1,ndim));
X2 = X * D; % scale the channels
[~,X3] = pca(X); % do a PCA rotation

% run the HMM
options = struct();
options.K = 4;
options.zeromean = 1;
options.verbose = 0;
options.useParallel = 0;
options.Gamma = initGamma_random(T,options.K,200);
% start always from the same solution - that is, no randomness

[hmm1,Gamma1] = hmmmar(X,T,options);
[hmm2,Gamma2] = hmmmar(X2,T,options);
[hmm3,Gamma3] = hmmmar(X3,T,options);

t = round((1:size(X,1))/1.33333)/60 ;
figure(1);
subplot(311); area(t,Gamma1); xlim([t(1) t(end)])
set(gca,'FontSize',20)
subplot(312); area(t,Gamma2); xlim([t(1) t(end)])
set(gca,'FontSize',20)
subplot(313); area(t,Gamma3); xlim([t(1) t(end)])
set(gca,'FontSize',20)

% Fig 2 panel B
% Now do same thing but across different selections of channels:
% how does PCA distortion relates to the spectra

nrep = 25; % number of repetitions
nchannels = 5:5:100; %[4:2:22 25:5:50 75 99];
ND = length(nchannels);

eigenspectra_mean = zeros(nrep,ND); % PCA concentration (see paper)
PCA_distortion = zeros(nrep,ND); % PCA distortion

for j1 = 1:ND
    ndim = nchannels(j1);
    for j = 1:nrep
        
        regions = randperm(size(X,2),ndim);
        X1 = X(:,regions); X1 = zscore(X1); %X = X + randn(size(X));
        
        T = size(X1,1);
        [~,X2,e] = pca(X1); e = cumsum(e)/sum(e);
        
        options = struct();
        options.K = 4;
        options.zeromean = 1;
        options.verbose = 0;
        options.useParallel = 0;
        
        [~,Gamma1] = hmmmar(X1,T,options);
        options2 = options; options2.Gamma = Gamma1; 
        % start from the previous solution state time courses
        [~,Gamma12] = hmmmar(X2,T,options2);
        
        eigenspectra_mean(j,j1) = mean(e);
        PCA_distortion(j,j1) = 1 - corr(Gamma1(:),Gamma12(:));
        
    end
    disp(['dim = ' num2str(j1)])
end

%% Simulations 3. Section "Simulated data experiments"

addpath(genpath('../HMM-MAR'))

nrep = 10; % number of repetitions 
nperm = 5000; % number of permutations for permutation testing

ttrial = 1000; ntrial = 100; N = ttrial * ntrial; K = 6; % data options
% the higher is change_threshold, the closer are the states; I tried 0.1 and 0.2
change_threshold = 0.2; 

% simulation options
stickyness_par = 25; % ratio of probability of staying in the same state vs changing states
KK = 4:8; npca = [5 10 20 30 40]; % HMM options: no. of states and no. of PCs

% Free energies
FE_mixpca = zeros(length(KK),length(npca),nrep);
FE_pca = zeros(length(KK),length(npca),nrep);
FE_std = zeros(length(KK),length(npca),nrep);

% how well we can predict the ground-truth state time courses?
r2_mixpca = zeros(length(KK),length(npca),nrep);
r2_pca = zeros(length(KK),length(npca),nrep);
r2_std = zeros(length(KK),length(npca),nrep);

% p-value (it's actually always very significant so I don't show in the paper)
pv_mixpca = zeros(length(KK),length(npca),nrep);
pv_pca = zeros(length(KK),length(npca),nrep);
pv_std = zeros(length(KK),length(npca),nrep);

for r = 1:nrep
    
    % load some fMRI data (HCP in Schaffer parcellation, 100 regions)
    load sub1_100Yeo.mat
    % generate the states (see paper)
    C0 = corr(D); ndim = size(D,2); K = 6;
    C = zeros(ndim,ndim,K);
    [V,D] = eig(C0); e = diag(D); e = e / sum(e);
    for k = 1:K
        ac_change = 0; chosen = [];
        while ac_change < change_threshold
            j = randsample(setdiff(1:ndim,chosen),1);
            if ac_change+e(j) > 1.5 * change_threshold, continue; end
            chosen = [chosen j];
            ac_change = ac_change + e(j);
        end
        V1 = V;
        for j = chosen
            v = V(:,j);
            V1(:,j) = v(randperm(length(v)));
        end
        C(:,:,k) = V1 * D * V1';
    end
    
    % generate state time courses (Gamma)
    T = ttrial * ones(ntrial,1);
    P = ones(K); P(eye(K)==1) = stickyness_par;
    for k = 1:K, P(k,:) = P(k,:) / sum(P(k,:)); end
    Pi = (1/K) * ones(1,K);
    Gamma_sim = simgamma(T,P,Pi);
    
    % generate data
    X_sim = zeros(N,ndim); done = false(N,1);
    for n = 1:ntrial
        tend = n * ttrial;
        t = 1 + (n-1) * ttrial;
        while t < tend
            this_state = find(Gamma_sim(t,:)==1);
            deltat = find(~Gamma_sim(t+1:tend,this_state),1);
            if isempty(deltat), deltat = tend - t + 1; end
            X_sim(t:t+deltat-1,:) = mvnrnd(zeros(deltat,ndim),C(:,:,this_state));
            done(t:t+deltat-1) = true;
            t = t + deltat;
        end
    end
    
    for ikk = 1:length(KK)
        for ipca = 1:length(npca)
            
            % standard HMM-Gaussian on PCA space
            options = struct();
            options.K = KK(ikk);
            options.zeromean = 1;
            options.pca = npca(ipca);
            options.cyc = 250;
            options.cyc = 50;
            options.initcyc = 5; 
            optins.initrep = 3;
            options.verbose = 0;
            [hmm_std,Gamma_std,~,~,~,~,fe] = hmmmar(X_sim,T,options);
            r2 = gamma_regress_cv (Gamma_std,Gamma_sim,T,nperm);
            r2_std(ikk,ipca,r) = r2(1); 
            FE_std(ikk,ipca,r) = fe(end); % hmmfe_ORIGSPACE(X_sim,T,hmm_std,Gamma_std,Xi); clear Xi
            
            % HMM-PCA
            options = struct();
            options.K = KK(ikk);
            options.zeromean = 1;
            options.lowrank = npca(ipca);
            options.cyc = 50;
            options.initcyc = 5; 
            optins.initrep = 3; 
            options.verbose = 0;
            [hmm_pca,Gamma_pca,~,~,~,~,fe] = hmmmar(X_sim,T,options);
            r2 = gamma_regress_cv (Gamma_pca,Gamma_sim,T,nperm);
            r2_pca(ikk,ipca,r) = r2(1);             
            FE_pca(ikk,ipca,r) = fe(end);
            
            % Mixture of PCAs
            options = struct();
            options.K = KK(ikk);
            options.zeromean = 1;
            options.lowrank = npca(ipca);
            options.id_mixture = 1;
            options.cyc = 50;
            options.initcyc = 5; 
            optins.initrep = 3;
            options.verbose = 0;
            [hmm_mixpca,Gamma_mixpca,~,~,~,~,fe] = hmmmar(X_sim,T,options);
            r2 = gamma_regress_cv (Gamma_mixpca,Gamma_sim,T,nperm);
            r2_mixpca(ikk,ipca,r) = r2(1);             
            FE_mixpca(ikk,ipca,r) = fe(end);
            
            disp([num2str(r) ' ' num2str(ikk) ' ' num2str(ipca)])
            save(['out/simulations_3_threshold_' num2str(change_threshold) '.mat'],'r2_std','r2_pca','r2_mixpca',...
                'FE_std','FE_pca','FE_mixpca','pv_std','pv_pca','pv_mixpca')
            
        end
    end
end

%% Real data experiments on HCP data

addpath(genpath('../HMM-MAR')) % assuming we are in the HMM-MAR directory

basedir = '/home/diegov/MATLAB/';
read_HCP
X = X - repmat(mean(X),size(X,1),1); ndim = size(X,2);

%% Run the HMM

npca = 24; % no. of PCAs: approx 80% of variance 
K = 12; % no. of states
nrep = 5; % number of repetitions, given that the inference is not deterministic
use_stochastic = 1; % use stochastic inference? (Vidaurre et al. 2018)

% FC: Functional connectivity for the inferred states
% FO: Fractional occupancies
FC_HMMGauss_cov = zeros(ndim,ndim,K,nrep);
FO_HMMGauss = zeros(N*4,K,nrep); 
FC_HMMPCA_cov = zeros(ndim,ndim,K,nrep);
FO_HMMPCA = zeros(N*4,K,nrep); 
FC_MixPCA_cov = zeros(ndim,ndim,K,nrep);
FO_MixPCA = zeros(N*4,K,nrep); 
FC_MixGauss_cov = zeros(ndim,ndim,K,nrep);
FO_MixGauss = zeros(N*4,K,nrep); 

% General HMM options
options = struct();
options.order = 0;
options.zeromean = 1;
options.K = K;
options.useParallel = 0;
options.cyc = 50;
options.standardise = 0;
options.verbose = 1;
options.initrep = 0; %options.initcyc = 10;

if use_stochastic
    options.BIGNbatch = 10;
    options.BIGtol = 1e-7;
    options.BIGcyc = 300;
    options.BIGundertol_tostop = 5;
    options.BIGforgetrate = 0.7;
    options.BIGbase_weights = 0.9;
end

% Specific options for each model
options_Gauss = options;
options_Gauss.pca = npca;
options_Gauss.covtype = 'full';
options_HMMPCA = options;
options_HMMPCA.lowrank = npca;
options_MixPCA = options;
options_MixPCA.id_mixture = 1;
options_MixPCA.lowrank = npca;
options_MixGauss = options;
options_MixGauss.id_mixture = 1;
options_MixGauss.pca = npca;
options_MixGauss.covtype = 'full';

% Cells with the models
hmm_HMMPCA = cell(nrep,1);
hmm_Gauss = cell(nrep,1);
hmm_MixPCA = cell(nrep,1);
hmm_MixGauss = cell(nrep,1);

for r = 1:nrep
    
    % run HMMs
    [hmm_Gauss{r},Gamma_Gauss] = hmmmar(X,T,options_Gauss);
    [hmm_HMMPCA{r},Gamma_HMMPCA] = hmmmar(X,T,options_HMMPCA);
    [hmm_MixPCA{r},Gamma_MixPCA] = hmmmar(X,T,options_MixPCA);
    [hmm_MixGauss{r},Gamma_MixGauss] = hmmmar(X,T,options_MixGauss);
    
    FO_HMMGauss(:,:,r) = getFractionalOccupancy (Gamma_Gauss,T,options_Gauss);
    FO_HMMPCA(:,:,r) = getFractionalOccupancy(Gamma_HMMPCA,T,options_HMMPCA);
    FO_MixPCA(:,:,r) = getFractionalOccupancy(Gamma_MixPCA,T,options_MixPCA);
    FO_MixGauss(:,:,r) = getFractionalOccupancy(Gamma_MixGauss,T,options_MixGauss);
    
    % get FC matrices for the states for each modality (do all in the same
    % way to minimise biases
    for k = 1:K
        
        Xw = X .* repmat(Gamma_Gauss(:,k),1,size(X,2)); 
        C1 = (Xw' * Xw) / sum(Gamma_Gauss(:,k));
        FC_HMMGauss_cov(:,:,k,r) = C1; 
        
        Xw = X .* repmat(Gamma_HMMPCA(:,k),1,size(X,2)); 
        C1 = (Xw' * Xw) / sum(Gamma_HMMPCA(:,k));
        FC_HMMPCA_cov(:,:,k,r) = C1;   
        
        Xw = X .* repmat(Gamma_MixPCA(:,k),1,size(X,2)); 
        C1 = (Xw' * Xw) / sum(Gamma_MixPCA(:,k));
        FC_MixPCA_cov(:,:,k,r) = C1;   
        
        Xw = X .* repmat(Gamma_MixGauss(:,k),1,size(X,2)); 
        C1 = (Xw' * Xw) / sum(Gamma_MixGauss(:,k));
        FC_MixGauss_cov(:,:,k,r) = C1;        
        
     end
    
    save('out/HCP_run.mat','FC_HMMGauss_cov','FC_HMMPCA_cov','FC_MixPCA_cov',...
        'FO_HMMGauss','FO_HMMPCA','FO_MixPCA',...
        'hmm_Gauss','hmm_HMMPCA','hmm_MixPCA' )
end
