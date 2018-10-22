% Code used in Vidaurre et al. (2017) PNAS
%
% Detailed documentation and further examples can be found in:
% https://github.com/OHBA-analysis/HMM-MAR
% This pipeline must be adapted to your particular configuration of files. 
%%%%%%%%%%%%%%%%%%%%%%%%%
%% SETUP THE MATLAB PATHS AND FILE NAMES
mydir = '/home/diegov/MATLAB/';
addpath(genpath([ mydir 'HMM-MAR']))

K = 12; % no. states
repetitions = 3; % to run it multiple times (keeping all the results)
DirData = [mydir 'data/HCP/TimeSeries/hippocampus_data/'];
DirOut = [mydir 'HCP/HMM/out/hippocampus/'];
TR = 0.72;  
use_stochastic = 1; % set to 1 if you have loads of data

N = 820; % no. subjects
f = cell(N,1); T = cell(N,1);
% each .mat file contains the data (ICA components) for a given subject, 
% in a matrix X of dimension (4800time points by 50 ICA components). 
% T{j} contains the lengths of each session (in time points)
for j=1:N
    f{j} = [DirData 's' num2str(j) '.mat'];
    load(f{j}); T{j} = [1200 1200 1200 1200];
end

options = struct();
options.K = K; % number of states 
options.order = 0; % no autoregressive components
options.zeromean = 0; % model the mean
options.covtype = 'full'; % full covariance matrix
options.Fs = 1/TR; 
options.verbose = 1;
options.standardise = 1;
options.inittype = 'HMM-MAR';
options.cyc = 500;
options.initcyc = 10;
options.initrep = 3;

% stochastic options
if use_stochastic
    options.BIGNbatch = round(N/30);
    options.BIGtol = 1e-7;
    options.BIGcyc = 500;
    options.BIGundertol_tostop = 5;
    options.BIGforgetrate = 0.7;
    options.BIGbase_weights = 0.9;
end

% We run the HMM multiple times
for r = 1:repetitions
    [hmm, Gamma, ~, vpath] = hmmmar(f,T,options);
    save([DirOut 'HMMrun_rep' num2str(r) '.mat'],'Gamma','vpath','hmm')
    disp(['RUN ' num2str(r)])
end

%% Pull out metastates

for r = 1:repetitions
    figure(r)
    load([DirOut 'HMMrun_rep' num2str(r) '.mat'],'Gamma','hmm')
    subplot(1,2,1) % Figure 2B
    GammaSessMean = squeeze(mean(reshape(Gamma,[1200 4 N K]),1));    
    GammaSubMean = squeeze(mean(GammaSessMean,1));
    [~,pca1] = pca(GammaSubMean','NumComponents',1);
    [~,ord] = sort(pca1); 
    imagesc(corr(GammaSubMean(:,ord))); colorbar
    subplot(1,2,2) % Figure 2A
    P = hmm.P;
    for j=1:K, P(j,j) = 0; P(j,:) = P(j,:) / sum(P(j,:));  end
    imagesc(P(ord,ord),[0 0.25]); colorbar
    axis square
    hold on
    for j=0:13
        plot([0 13] - 0.5,[j j] + 0.5,'k','LineWidth',2)
        plot([j j] + 0.5,[0 13] - 0.5,'k','LineWidth',2)
    end
    hold off
end
    
   