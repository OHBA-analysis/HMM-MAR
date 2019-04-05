% Code used in Vidaurre et al. (2016) NeuroImage
%
% Detailed documentation and further examples can be found in:
% https://github.com/OHBA-analysis/HMM-MAR
% This pipeline must be adapted to your particular configuration of files. 
%%%%%%%%%%%%%%%%%%%%%%%%%
%% SETUP THE MATLAB PATHS AND FILE NAMES
mydir = '/home/diegov/MATLAB'; % adapt to yours
addpath(genpath(mydir))

load('~/Work/data/HCP/MEG-Task/data_task_mot_rh.mat')
% This contains source reconstructed MEG data with two sources in the
% primary motor cortex (left and right). 
% The data correspond to the motor task MEG data from the HCP,
% which is downloadable from the HCP site.
% The file data_task_mot_rh is struct with a 'data' field, a cell with 52 elements, 
% each of which corresponds to one subject and has dimensions (time by 2);
% and a 'T' field, a cell with 52 elements, 
% each of which is a vector with the length of each epoch/session (in time points) 
% for that particular subject, such that size(X{n},1) == sum(T{n}) 

% The data used in the NeuroImage 2016 paper was on a different data set,
% collected in Nottingham. The results are analogous. The data used in this
% script also corresponds to Vidaurre et al. (2017) NeuroImage. 

data = data_vect_t_group.data;
T = data_vect_t_group.T;
N = length(T);

clear data_vect_t_group
K = 3; Hz = 100; order = 5; 

options = struct();
% Specific stochastic inference options (Vidaurre et al, NeuroImage 2017)
options.K = K;
options.BIGNinitbatch = 9;
options.BIGNbatch = 4;
options.BIGtol = 1e-7;
options.BIGcyc = 500;
options.BIGundertol_tostop = 5;
options.BIGforgetrate = 0.7;
options.BIGbase_weights = 0.9;
% HMM-MAR options
options.covtype = 'diag';
options.order = order; 
options.zeromean = 1;
options.tol = 1e-7;
options.cyc = 50;
options.DirichletDiag = 1000;
options.initcyc = 20;
options.initrep = 3;
options.Fs = 100;
options.p = 0;
options.standardise = 1;

%%

[hmm, Gamma,~,~,~,~,fehist] = hmmmar(data,T,options);
Gamma = single(Gamma);
spectramar = hmmspectramar(data,T,hmm,Gamma,options);
spectramt = hmmspectramt(data,T,Gamma,options);

%% HMM-MAR on synthetic data 
% generating surrogate data to be able to say: states lock to stimulus

Tmat = [];
for n = 1:length(T), Tmat = [Tmat; T{n}']; end
datasim = simhmmmar(Tmat,hmm);
datasim2 = cell(N,1);
acc = 0;
for n = 1:N
    ind = acc + (1:sum(T{n}));
    datasim2{n} = datasim(ind,:); 
    acc = acc + length(ind);
end
datasim = datasim2; clear datasim2

[hmmsim,Gammasim] = hmmmar(datasim,T,options);
spectramar_sim = hmmspectramar(datasim,T,hmmsim,Gammasim,options);


%% Point process corresponding to the onset of the states 
% This is specific to the HCP data, where the data was chopped in trials of
% 221 time points each. It can be easily adapted to continuous data (of the
% likes of the NeuroImage 2016 paper) 

c = 0 ;
Onsets = cell(N,1);
for n = 1:N
    Onsets{n} = cell(length(T{n}),hmm.train.K);
    for j = 1:length(T{n})
        t = (1:T{n}(j)-order) + c; c = c + length(t);
        [~,g] = max(Gamma(t,:),[],2); % better use the Viterbi path 
        onsets = getStateOnsets(g,T{n}(j),Hz,hmm.train.K);
        for k = 1:K
            Onsets{n}{j,k} = onsets{1,k};
        end
    end
end

ttrial = T{1}(1);
PSTH = zeros(ttrial-order,N,K);
for k = 1:4
    for n = 1:N
        PSTHn = zeros(ttrial-order,length(T{n}));
        for j = 1:length(T{n})
            PSTHn(round(Onsets{n}{j,k}*100),j) = 1;
        end
        PSTH(:,n,k) = mean(PSTHn,2);
    end
end

t1 = (ttrial-order);
Onset_pval = ones(t1,1); 
Onset_st =  ones(t1,1);

for t = 1:t1
    [~,k] = max(mean(squeeze(PSTH(t,:,:))));
    tmp = squeeze(PSTH(t,:,setdiff(1:K,k)));
    [~,k2] = max(mean(tmp));
    Onset_st(t) = k;
    d1 = squeeze(PSTH(t,:,k))';
    d2 = tmp(:,k2);
    Onset_pval(t) = permtestdiff_aux(d1,d2,10000,[],1);
    [t k k2]
end

figure
plot((1:t1)/Hz,-log(Onset_pval),'Color','k','LineWidth',2) % - log(pval), the higher the better
xlim([1/Hz t1/Hz]);  