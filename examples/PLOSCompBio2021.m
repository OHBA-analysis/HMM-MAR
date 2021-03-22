%% Simulations 1. Section "Loss of sensitivity" 
% Section 3.1

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



%% Simulations 2. Section "Bias towards low-order PCA components"
% Section 3.2

addpath(genpath('../HMM-MAR'))

datadir = '/Users/admin/Work/data/HCP/TimeSeries/';
% load some fMRI data (HCP in Schaffer parcellation, 100 regions)
load([datadir 'sub1_100Yeo.mat']); X = D; clear D

%%% Fig 3 panel A

ndim = 4; regions = randperm(size(X,2),ndim); % get 4 random regions
X = X(1:1200,regions); X = zscore(X); % get a chunk of data

T = size(X,1);
D = diag(randn(1,ndim));
X2 = X * D; % scale the channels
[~,X3] = pca(X); % do a PCA rotation

options = struct();
options.K = 4;
options.zeromean = 1;
options.verbose = 0;
options.useParallel = 0;
options.Gamma = initGamma_random(T,options.K,200);
% start always from the same solution - that is, no randomness

% run the HMM
[~,Gamma1] = hmmmar(X,T,options);
[~,Gamma2] = hmmmar(X2,T,options);
[~,Gamma3] = hmmmar(X3,T,options);

t = round((1:size(X,1))/1.33333)/60 ;
figure(1);
subplot(311); area(t,Gamma1); xlim([t(1) t(end)])
set(gca,'FontSize',20)
subplot(312); area(t,Gamma2); xlim([t(1) t(end)])
set(gca,'FontSize',20)
subplot(313); area(t,Gamma3); xlim([t(1) t(end)])
set(gca,'FontSize',20)


[corr(Gamma1(:),Gamma2(:))  corr(Gamma1(:),Gamma3(:)) ]

%%% Fig 3 panel B
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
        [~,Gamma12] = hmmmar(X2,T,options2);
        
        eigenspectra_mean(j,j1) = mean(e);
        PCA_distortion(j,j1) = 1 - corr(Gamma1(:),Gamma12(:));
        
    end
    disp(['dim = ' num2str(j1)])
end

%% Fig 3 panel C

nrep = 50; C = zeros(nrep,2);

for r = 1:nrep
    
    ndim = 4; regions = randperm(size(X,2),ndim); % get 4 random regions
    X = X(1:1200,regions); X = zscore(X); % get a chunk of data
    
    T = size(X,1);
    D = diag(randn(1,ndim));
    [~,X3] = pca(X); % do a PCA rotation
    
    % HMM+Gaussian
    options = struct();
    options.K = 4;
    options.zeromean = 1;
    options.verbose = 0;
    options.useParallel = 0;
    options.Gamma = initGamma_random(T,options.K,200);
    % start always from the same solution - that is, no randomness
    
    % HMM-PCA
    options_hmmpca = options;
    options_hmmpca.lowrank = 2;
    
    [~,Gamma11] = hmmmar(X,T,options_hmm);
    [~,Gamma12] = hmmmar(X3,T,options_hmm);
    
    [~,Gamma21] = hmmmar(X,T,options_hmmpca);
    [~,Gamma22] = hmmmar(X3,T,options_hmmpca);
    
    C(r,1) = corr(Gamma11(:),Gamma12(:)); % HMM-Gauss
    C(r,2) = corr(Gamma21(:),Gamma22(:)); % HMM-PCA
    
        
end

% Plot
figure(3);clf(3)
hold on
bar(1,mean(C(:,1)),'FaceColor',[0.3 0.3 1])
bar(2,mean(C(:,2)),'FaceColor',[1 0.5 0.5])
scatter(0.1*randn(size(C,1),1) + ones(size(C,1),1),C(:,1),80,'b','filled')
scatter(0.1*randn(size(C,1),1) + 2*ones(size(C,1),1),C(:,2),80,'r','filled')
scatter(0.1*randn(size(C,1),1) + ones(size(C,1),1),C(:,1),80,'b','filled')
scatter(0.1*randn(size(C,1),1) + 2*ones(size(C,1),1),C(:,2),80,'r','filled')
hold off


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

% Free energies (not shown in paper)
FE_mixpca = zeros(length(KK),length(npca),nrep);
FE_pca = zeros(length(KK),length(npca),nrep);
FE_std = zeros(length(KK),length(npca),nrep);

% how well we can predict the ground-truth state time courses?
r2_mixpca = zeros(length(KK),length(npca),nrep);
r2_pca = zeros(length(KK),length(npca),nrep);
r2_std = zeros(length(KK),length(npca),nrep);

% cross-validated likelihood
cv_mixpca = zeros(length(KK),length(npca),nrep);
cv_pca = zeros(length(KK),length(npca),nrep);
cv_std = zeros(length(KK),length(npca),nrep);

for r = 1:nrep
    
    % load some fMRI data (HCP in Schaffer parcellation, 100 regions)
    load sub1_100Yeo.mat
    % generate the states (see paper)
    C0 = corr(D); ndim = size(D,2); K = 6;
    C = zeros(ndim,ndim,K);
    [V,D] = eig(C0); 
    e = diag(D); e = e / sum(e);
    Vk = cell(K,1); Dk = cell(K,1);
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
                Vk{k} = V1;
        Dk{k} = D;
    end
    
    % generate state time courses (Gamma)
    
    Gamma_sim = zeros(500,200,K);
    for j = 1:200, Gamma_sim(:,j,randi(K)) = 1; end
    Gamma_sim = reshape(Gamma_sim,500*200,K);
    T = ttrial * ones(ntrial,1);
    
    %T = ttrial * ones(ntrial,1);
    %P = ones(K); P(eye(K)==1) = stickyness_par;
    %for k = 1:K, P(k,:) = P(k,:) / sum(P(k,:)); end
    %Pi = (1/K) * ones(1,K);
    %Gamma_sim = simgamma(T,P,Pi);
    
    % generate data
    X_sim = zeros(N,ndim); 
    for k = 1:K
       ind = Gamma_sim(:,k) == 1; 
       latent = randn(sum(ind),ndim);
       X_sim(ind,:) = latent * sqrt(Dk{this_state}) * Vk{this_state}' + ...
                1e-6 * randn(sum(ind),ndim);
    end
    
    for ikk = 1:length(KK)
        for ipca = 1:length(npca)
            
            % standard HMM-Gaussian on PCA space
            options = struct();
            options.K = KK(ikk);
            options.zeromean = 1;
            options.pca = npca(ipca);
            options.cyc = 50;
            options.initcyc = 5; 
            options.initrep = 3;
            options.verbose = 0;
            [hmm_std,Gamma_std,~,~,~,~,fe] = hmmmar(X_sim,T,options);
            options.cvfolds = 2;
            mcv = cvhmmmar(X_sim,T,options);
            r2 = gamma_regress_cv (Gamma_std,Gamma_sim,T,nperm);
            r2_std(ikk,ipca,r) = r2(1); 
            FE_std(ikk,ipca,r) = fe(end); 
            cv_std(ikk,ipca,r) = mcv;
            
            % HMM-PCA
            options = struct();
            options.K = KK(ikk);
            options.zeromean = 1;
            options.lowrank = npca(ipca);
            options.cyc = 50;
            options.initcyc = 5; 
            options.initrep = 3; 
            options.verbose = 0;
            [hmm_pca,Gamma_pca,~,~,~,~,fe] = hmmmar(X_sim,T,options);
            options.cvfolds = 2;
            mcv = cvhmmmar(X_sim,T,options);
            r2 = gamma_regress_cv (Gamma_pca,Gamma_sim,T,nperm);
            r2_pca(ikk,ipca,r) = r2(1);             
            FE_pca(ikk,ipca,r) = fe(end);
            cv_pca(ikk,ipca,r) = mcv;
            
            % Mixture of PCAs
            options = struct();
            options.K = KK(ikk);
            options.zeromean = 1;
            options.lowrank = npca(ipca);
            options.id_mixture = 1;
            options.cyc = 50;
            options.initcyc = 5; 
            options.initrep = 3;
            options.verbose = 0;
            [hmm_mixpca,Gamma_mixpca,~,~,~,~,fe] = hmmmar(X_sim,T,options);
            options.cvfolds = 2;
            mcv = cvhmmmar(X_sim,T,options);
            r2 = gamma_regress_cv (Gamma_mixpca,Gamma_sim,T,nperm);
            r2_mixpca(ikk,ipca,r) = r2(1);             
            FE_mixpca(ikk,ipca,r) = fe(end);
            cv_mixpca(ikk,ipca,r) = mcv;
            
            disp([num2str(r) ' ' num2str(ikk) ' ' num2str(ipca)])
            
            save(['out/simulations_3_threshold_' num2str(change_threshold) '.mat'],...
                'r2_std','r2_pca','r2_mixpca',...
                'FE_std','FE_pca','FE_mixpca','pv_std','pv_pca','pv_mixpca',...
                'cv_std','cv_pca','cv_mixpca')
            
            save(['out/simulations_3_threshold_' num2str(change_threshold) '.mat'],...
                'r2_std','r2_pca','r2_mixpca',...
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


%% Gather behaviourals

datadir = '/home/diegov/MATLAB/data/HCP/';
%datadir = '~/Work/data/HCP/';
vars1 = dlmread([ datadir 'scripts1200/vars/vars.txt'] ,' ');
vars = dlmread([ datadir 'scripts900/vars.txt'] ,' ');
p = size(vars,2);
%twins = dlmread([ datadir 'scripts1200/vars/twins.txt'],' ');
twins = dlmread([ datadir 'scripts900/twins.txt'],' ');
twins = twins(2:end,2:end);

head1 = load('../experiments_HCP1200/headers_with_category.mat');
head1=head1.headers;
fid = fopen('column_headers.txt');
head2 = textscan(fid,'%s');
head2 = head2{1}; head2(1) = []; % because it takes the 1st row in two elements
p2 = length(head2);

grotKEEP = true(size(vars,1),1);
grotKEEP(find(vars(:,1)==376247))=0; % remove subjects that PALM can't cope with
grotKEEP(find(vars(:,1)==168240))=0; % remove subjects that PALM can't cope with
grotKEEP(find(vars(:,1)==122418))=0; % remove subjects that PALM can't cope with
twins = twins(grotKEEP,grotKEEP);
confounds = [3 8]; % sex, motion
conf = zscore(vars(grotKEEP,confounds));
keep = true(1,p);
% for BV = 1:p
%     Y=vars(:,BV);
%     if sum(~isnan(Y))<500, keep(BV) = false; continue; end
% end

Yall = []; behdomain = [];

% 1. Demographical
to_use = false(p2,1);
to_use(strcmp(head1(:,2),'Demographics')) = true;
to_use(1) = false; to_use(2) = false; % remove ID and recon
to_use(3) = false; to_use(8) = false; % remove sex and motion
IND = false(p,1); to_use = find(to_use);
for j = 1:length(to_use)
    jj=find(strcmp(head2, head1{to_use(j),1}));
    if ~isempty(jj), IND(jj) = true; end
end
Y = vars(grotKEEP,IND);
Yall = [Yall Y];
behdomain = [behdomain 1*ones(1,size(Y,2))];

% 2. Intelligence
to_use = false(p,1);
to_use(strcmp(head1(:,2),'Fluid Intelligence')) = true;
to_use(strcmp(head1(:,2),'Language/Reading')) = true;
to_use(strcmp(head1(:,2),'Language/Vocabulary')) = true;
to_use(strcmp(head1(:,2),'Processing Speed')) = true;
to_use(strcmp(head1(:,2),'Spatial Orientation')) = true;
to_use(strcmp(head1(:,2),'Sustained Attention')) = true;
to_use(strcmp(head1(:,2),'Verbal Episodic Memory')) = true;
to_use(strcmp(head1(:,2),'Working Memory')) = true;
to_use(strcmp(head1(:,2),'Episodic Memory')) = true;
to_use(strcmp(head1(:,2),'Executive Function/Cognitive Flexibility')) = true;
to_use(strcmp(head1(:,2),'Executive Function/Inhibition')) = true;
to_use(strcmp(head1(:,2),'Alertness')) = true;
to_use(510) = true; % Language_Task_Acc
to_use(518) = true; % Relational_Task_Acc
to_use(545) = true; % Working memory task acc
IND = false(p,1); to_use = find(to_use);
for j = 1:length(to_use)
    jj=find(strcmp(head2, head1{to_use(j),1}));
    if ~isempty(jj), IND(jj) = true; end
end
Y = vars(grotKEEP,IND);
Yall = [Yall Y];
behdomain = [behdomain 2*ones(1,size(Y,2))];

% 5. Affective variables
to_use = false(p,1);
to_use(strcmp(head1(:,2),'Negative Affect')) = true;
to_use(strcmp(head1(:,2),'Psychological Well-being')) = true;
to_use(strcmp(head1(:,2),'Social Relationships')) = true;
to_use(strcmp(head1(:,2),'Stress and Self-efficacy')) = true;
IND = false(p,1); to_use = find(to_use);
for j = 1:length(to_use)
    jj=find(strcmp(head2, head1{to_use(j),1}));
    if ~isempty(jj), IND(jj) = true; end
end
Y = vars(grotKEEP,IND);
Yall = [Yall Y];
behdomain = [behdomain 5*ones(1,size(Y,2))];

% 6. Personality
to_use = false(p,1);
to_use(strcmp(head1(:,2),'Personality')) = true;
IND = false(p,1); to_use = find(to_use);
for j = 1:length(to_use)
    jj=find(strcmp(head2, head1{to_use(j),1}));
    if ~isempty(jj), IND(jj) = true; end
end
Y = vars(grotKEEP,IND);
Yall = [Yall Y];
behdomain = [behdomain 6*ones(1,size(Y,2))];

% 8. Sleep
to_use = false(p,1);
to_use(strcmp(head1(:,2),'Sleep')) = true;
IND = false(p,1); to_use = find(to_use);
for j = 1:length(to_use)
    jj=find(strcmp(head2, head1{to_use(j),1}));
    if ~isempty(jj), IND(jj) = true; end
end
Y = vars(grotKEEP,IND);
Yall = [Yall Y];
behdomain = [behdomain 8*ones(1,size(Y,2))];


%% predicting with elastic net using all runs
% uses nets_predict5, from here: https://github.com/vidaurre/NetsPredict

K =  size(FO_HMMGauss,2);
addpath('../PredictionDV/NetsPredict/') 
X1 = squeeze(mean(reshape(FO_HMMGauss,[4 820 K*5]),1)); X1 = X1(grotKEEP,:);
X2 = squeeze(mean(reshape(FO_HMMPCA,[4 820 K*5]),1)); X2 = X2(grotKEEP,:);
X3 = squeeze(mean(reshape(FO_MixGauss,[4 820 K*5]),1)); X3 = X3(grotKEEP,:);
X4 = squeeze(mean(reshape(FO_MixPCA,[4 820 K*5]),1)); X4 = X4(grotKEEP,:);

Tall = cell2mat(T); Tall = Tall(:);
MaxFO = zeros(5,4);
for r = 1:5
    MaxFO(r,1) = mean(getMaxFractionalOccupancy(FO_HMMGauss(:,:,r),Tall));
    MaxFO(r,2) = mean(getMaxFractionalOccupancy(FO_HMMPCA(:,:,r),Tall));
    MaxFO(r,3) = mean(getMaxFractionalOccupancy(FO_MixGauss(:,:,r),Tall));
    MaxFO(r,4) = mean(getMaxFractionalOccupancy(FO_MixPCA(:,:,r),Tall));
end; clear Tall

parameters_prediction = struct();
parameters_prediction.Method = 'ridge';
parameters_prediction.CVscheme = [10 10];

q = size(Yall,2);
COD = zeros(4,q); CORR = zeros(4,q); 

for jj = 1:q
    ind = ~isnan(Yall(:,jj));
    stats1 = ...
        nets_predict5(Yall(ind,jj),X1(ind,:),'Gaussian',...
        parameters_prediction,twins(ind,ind),[],conf(ind,:));
    stats2 = ...
        nets_predict5(Yall(ind,jj),X2(ind,:),'Gaussian',...
        parameters_prediction,twins(ind,ind),[],conf(ind,:));
    stats3 = ...
        nets_predict5(Yall(ind,jj),X3(ind,:),'Gaussian',...
        parameters_prediction,twins(ind,ind),[],conf(ind,:));
    stats4 = ...
        nets_predict5(Yall(ind,jj),X4(ind,:),'Gaussian',...
        parameters_prediction,twins(ind,ind),[],conf(ind,:));
    
    COD(1,jj) = stats1.cod; COD(2,jj) = stats2.cod;
    COD(3,jj) = stats3.cod; COD(4,jj) = stats4.cod;
    CORR(1,jj) = stats1.corr; CORR(2,jj) = stats2.corr;
    CORR(3,jj) = stats3.corr; CORR(4,jj) = stats4.corr;
    
    disp([num2str(jj) ' of ' num2str(q)])
end

save('out/prediction_HCP.mat','COD','CORR','-append') 

%% Grid run over PCA and K

addpath(genpath('../HMM-MAR')) % assuming we are in the HMM-MAR directory

basedir = '/home/diegov/MATLAB/';
read_HCP
%X = X - repmat(mean(X),size(X,1),1); 
ndim = 50;

use_stochastic = 1;

options = struct();
options.order = 0;
options.zeromean = 1;
options.useParallel = 1;
options.cyc = 100;
options.standardise = 0;
options.verbose = 1;
options.initrep = 0; %options.initcyc = 10;

if use_stochastic
    options.BIGNbatch = round(N/30);
    options.BIGtol = 1e-7;
    options.BIGcyc = 150;
    options.BIGundertol_tostop = 5;
    options.BIGforgetrate = 0.7;
    options.BIGbase_weights = 0.9;
    options.BIGinitcyc = 10;
    options.BIGinitStrategy = 2;
end

options_Gauss = options;
options_Gauss.covtype = 'full';
options_HMMPCA = options;
options_MixPCA = options;
options_MixPCA.id_mixture = 1;

KK = 6:2:16; npca = 12:4:32; % HMM options
nrep = 3;

hmm_HMMPCA = cell(length(KK),length(npca),nrep);
hmm_Gauss = cell(length(KK),length(npca),nrep);
hmm_MixPCA = cell(length(KK),length(npca),nrep);

FO_HMMGauss = cell(length(KK),length(npca));
FO_HMMPCA = cell(length(KK),length(npca));
FO_MixPCA = cell(length(KK),length(npca));

        
for ikk = 1:length(KK)
    for ipca = 1:length(npca)
        
        K = KK(ikk);
        options_Gauss.K = K;
        options_HMMPCA.K = K;
        options_MixPCA.K = K;
        options_Gauss.pca = npca(ipca);
        options_HMMPCA.lowrank = npca(ipca);
        options_MixPCA.lowrank = npca(ipca);
        
        FO_HMMGauss{ikk,ipca} = zeros(N*4,K,nrep);
        FO_HMMPCA{ikk,ipca} = zeros(N*4,K,nrep);
        FO_MixPCA{ikk,ipca} = zeros(N*4,K,nrep);
        
        for r = 1:nrep
            
            [hmm_Gauss{ikk,ipca,r},Gamma_std] = hmmmar(X,T,options_Gauss);
            [hmm_HMMPCA{ikk,ipca,r},Gamma_pca] = hmmmar(X,T,options_HMMPCA);
            [hmm_MixPCA{ikk,ipca,r},Gamma_MixPCA] = hmmmar(X,T,options_MixPCA);
            
            FO_HMMGauss{ikk,ipca}(:,:,r) = getFractionalOccupancy (Gamma_std,cell2mat(T),options_Gauss);
            FO_HMMPCA{ikk,ipca}(:,:,r) = getFractionalOccupancy(Gamma_pca,cell2mat(T),options_HMMPCA);
            FO_MixPCA{ikk,ipca}(:,:,r) = getFractionalOccupancy(Gamma_MixPCA,cell2mat(T),options_MixPCA);
            
            disp([ num2str(r) ': ' num2str(ikk) ' ' num2str(ipca) ])
            
            save('out/HCP_run_GRID_manyreps.mat',...
                'FO_HMMGauss','FO_HMMPCA','FO_MixPCA',...
                'hmm_Gauss','hmm_MixPCA','hmm_HMMPCA')
            
        end
        
    end
end

%% predicting with elastic net using all runs
% uses nets_predict5, from here: https://github.com/vidaurre/NetsPredict

load('out/HCP_run_GRID_manyreps.mat')
addpath('../PredictionDV/NetsPredict/')

q = size(Yall,2); nrep = size(FO_HMMGauss{1,1},3);
COD = zeros(4,q,size(FO_HMMGauss,1),size(FO_HMMGauss,2)); 
CORR = zeros(4,q,size(FO_HMMGauss,1),size(FO_HMMGauss,2)); 
MaxFO = zeros(820*4,size(FO_HMMGauss,1),size(FO_HMMGauss,2),nrep,3);

for ikk = 1:size(FO_HMMGauss,1)
    for ipca = 1:size(FO_HMMGauss,2)
        
        K =  size(FO_HMMGauss{ikk,ipca},2);
        X1 = squeeze(mean(reshape(FO_HMMGauss{ikk,ipca},[4 820 K*nrep]),1)); 
        X1 = X1(grotKEEP,:);
        X2 = squeeze(mean(reshape(FO_HMMPCA{ikk,ipca},[4 820 K*nrep]),1)); 
        X2 = X2(grotKEEP,:);
        X3 = squeeze(mean(reshape(FO_MixPCA{ikk,ipca},[4 820 K*nrep]),1)); 
        X3 = X3(grotKEEP,:);
        
        for r = 1:nrep
            MaxFO(:,ikk,ipca,r,1) = max(FO_HMMGauss{ikk,ipca}(:,:,r),[],2);
            MaxFO(:,ikk,ipca,r,2) = max(FO_HMMPCA{ikk,ipca}(:,:,r),[],2);
            MaxFO(:,ikk,ipca,r,3) = max(FO_MixPCA{ikk,ipca}(:,:,r),[],2);
        end
        
        parameters_prediction = struct();
        parameters_prediction.Method = 'ridge';
        parameters_prediction.CVscheme = [10 10];
        
            
        for jj = 1:q
            ind = ~isnan(Yall(:,jj));
            stats1 = ...
                nets_predict5(Yall(ind,jj),X1(ind,:),'Gaussian',...
                parameters_prediction,twins(ind,ind),[],conf(ind,:));
            stats2 = ...
                nets_predict5(Yall(ind,jj),X2(ind,:),'Gaussian',...
                parameters_prediction,twins(ind,ind),[],conf(ind,:));
            stats3 = ...
                nets_predict5(Yall(ind,jj),X3(ind,:),'Gaussian',...
                parameters_prediction,twins(ind,ind),[],conf(ind,:));
            
            COD(1,jj,ikk,ipca) = stats1.cod; 
            COD(2,jj,ikk,ipca) = stats2.cod;
            COD(3,jj,ikk,ipca) = stats3.cod; 
            CORR(1,jj,ikk,ipca) = stats1.corr; 
            CORR(2,jj,ikk,ipca) = stats2.corr;
            CORR(3,jj,ikk,ipca) = stats3.corr; 
            
            disp([num2str(ikk) ', ' num2str(ipca) '; ' num2str(jj) ' of ' num2str(q)])
        end
        
        save('out/HCP_run_GRID_manyreps_BEH.mat','COD','CORR','MaxFO')
        
    end
    
end


%% Robustness of the solutions

load('out/HCP_run.mat')


C1 = []; C2 = [];
for r1 = 1:5
    for r2 = 1:5
        if r1==r2, continue; end
        r = gamma_regress_cv (FO_HMMGauss(:,1:end-1,r1),FO_HMMGauss(:,1:end-1,r2),...
            ones(3280,1),100);
        C1 = [C1; mean(r)];
        r = gamma_regress_cv (FO_HMMPCA(:,1:end-1,r1),FO_HMMPCA(:,1:end-1,r2),...
            ones(3280,1),100);
        C2 = [C2; mean(r)];      
    end
end

% Plot 

figure(6);clf(6)

hold on
bar(1,mean(1-C1),'FaceColor',[0.3 0.3 1])
scatter(1 + 0.05*randn(size(C1,1),1),1-C1,60,'b','filled')
bar(2,mean(1-C2),'FaceColor',[1 0.5 0.5])
scatter(2 + 0.05*randn(size(C2,1),1),1-C2,60,'r','filled')
hold off
set(gca,'Xtick',[],'FontSize',20)
xlim([0.5 2.5]);ylim([0.975 0.99])


