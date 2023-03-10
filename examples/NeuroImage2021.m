% Code used in Vidaurre et al. (2021) NEuroImage
%
% This script assumes that all the preprocessing has been done and 
% that we have in memory: 
%
% - The rs fMRI data, including f, the file names with the rs-fMRI data, and
% T, the length of each session 
% (T is a cell, and T{subject} = [1200 1200 1200 1200] in the
% case of the HCP, because there are 1200 time points per session.)
% - The behavioural variables in vars
% - The family structure in twins
% - The confounds in conf (eg motion,sex)
% - The structural data in cell Anatomy (three elements, one per each type of structural) 
%
% The pipeline is
%
% 1. Running the HMM with the function hmmmar and computing the
% dual-estimated HMM models with hmmdual
%
% 2. Compute the distance matrices in HMM space as well as for the static
% FC. This is done respectively with hmm_kl() on the dual estimated models.
% For the static FC matrices, this is done with the wishart_kl function. 
%
% 3. Compute the predictions on the structural data, also based on distance
% matrices. These are just Euclidean distances. The prediction is done
% through the predictPhenotype.m function. Then, save the residuals of
% these predictions as deconfounded behavioural variables. 
%
% 4. Run the predictions using the static FC distance matrices on the raw
% or structure-deconfounded behavioural variables, using predictPhenotype.m
%
% 5. Similarly, run the predictions using each of the HMM distance matrices on the raw
% or structure-deconfounded behavioural variables, using predictPhenotype.m

mydir = '/home/diegov/MATLAB/'; % set to your directory
addpath(genpath([ mydir 'HMM-MAR'])) % HMM repository

ICAdim = 50; % number of ICA components (25,50,100,etc)
K = 8; % no. states
covtype = 'full'; % type of covariance matrix
repetitions = 5; % to run it multiple times (keeping all the results)

DirData = [mydir 'data/HCP/TimeSeries/group1200/3T_HCP1200_MSMAll_d' num2str(ICAdim) '_ts2/'];
DirOut = [mydir 'experiments_HCP1200/hmms/ICA' num2str(ICAdim) '/K' num2str(K)];
% We will save here the distance matrices

TR = 0.75; N = length(f); 

%% Run the HMMs (5 repetitions, with states characterised by a covariance matrix)

options = struct();
options.K = K; % number of states
options.order = 0; % no autoregressive components
options.covtype = covtype;
options.zeromean = zeromean;
options.Fs = 1/TR;
options.standardise = 1;
options.DirichletDiag = 10;
options.dropstates = 0;
options.cyc = 50;
options.initcyc = 5;
options.initrep = 3;
options.verbose = 0;
% stochastic options
options.BIGNbatch = round(N/30);
options.BIGtol = 1e-7;
options.BIGcyc = 100;
options.BIGundertol_tostop = 5;
options.BIGforgetrate = 0.7;
options.BIGbase_weights = 0.9;

options_singlesubj = struct();
options_singlesubj.K = K; % number of states
options_singlesubj.order = 0; % no autoregressive components
options_singlesubj.zeromean = zeromean; % don't model the mean
options_singlesubj.covtype = covtype;
options_singlesubj.Fs = 1/TR;
options_singlesubj.standardise = 1;
options_singlesubj.DirichletDiag = 10;
options_singlesubj.dropstates = 0;
options_singlesubj.cyc = 100;
options_singlesubj.verbose = 0;

options_singlesubj_dr = options_singlesubj;
options_singlesubj_dr.updateGamma = 0;

% We run the HMM multiple times
for r = 1:repetitions
    
    % Run the HMM at the group level and get some statistics
    % (eg fractional occupancy)
    [hmm,Gamma] = hmmmar(f,T,options);
    FOgroup = zeros(N,K); % Fractional occupancy
    meanActivations = zeros(ICAdim,K); % maps
    for j=1:N
        ind = (1:4800) + 4800*(j-1);
        FOgroup(j,:) = mean(Gamma(ind,:));
        X = zscore(dlmread(f{j}));
        for k = 1:K
            meanActivations(:,k) = meanActivations(:,k) + ...
                sum(X .* repmat(Gamma(ind,k),1,ICAdim))';
        end
    end
    meanActivations = meanActivations ./ repmat(sum(Gamma),ICAdim,1);
    switchingRate = getSwitchingRate(Gamma,T,options);
    maxFO = getMaxFractionalOccupancy(Gamma,T,options);
    
    % Subject specific stuff (dual-estimation)
    options_singlesubj.hmm = hmm;
    FOdual = zeros(N,K);
    parfor j = 1:N
        X = dlmread(f{j});
        % dual-estimation
        [HMMs_dualregr{j},Gammaj] = hmmdual(X,T{j},hmm);
        for k = 1:K
            HMMs_dualregr{j}.state(k).prior = [];
        end
        FOdual(j,:) = mean(Gammaj);
    end
    
    save([DirOut 'HMMs_r' num2str(r) '_GROUP.mat'],...
        'hmm','FOgroup','meanActivations','switchingRate','maxFO')
    save([DirOut 'HMMs_r' num2str(r)  '.mat'],...
        'HMMs','HMMs_dualregr','FO','FOdual')
    
    disp(['Repetition ' num2str(r) ])
end

%% Create distance matrices between  models 

% between HMMs
DistHMM = zeros(N,N,5); % subj x subj x repetitions
for r = 1:repetitions
    out = load([DirOut 'HMMs_r' num2str(r) '.mat']);
    for n1 = 1:N-1
        for n2 = n1+1:N
            % FO is contained in TPC; TPC is contained in HMM
            DistHMM(n1,n2,r) = (hmm_kl(out.HMMs_dualregr{n1},out.HMMs_dualregr{n2}) ...
                + hmm_kl(out.HMMs_dualregr{n2},out.HMMs_dualregr{n1}))/2;
            DistHMM(n2,n1,r) = DistHMM(n1,n2,r);
        end
    end
    disp(num2str(r))
end

% between static FC matrices
DistStatic = zeros(N);
for n1 = 1:N-1
    for n2 = n1+1:N
        DistStatic(n1,n2) = ( wishart_kl(V(:,:,n1),V(:,:,n2),sum(T{n1}),sum(T{n2})) + ...
            wishart_kl(V(:,:,n2),V(:,:,n1),sum(T{n2}),sum(T{n1})) ) /2;
        DistStatic(n2,n1) = DistStatic(n1,n2);
    end
end

save(['KLdistances_ICA' num2str(ICAdim) '.mat'],'DistMat','DistStatic');


%% Predictions of behaviour using structurals
% The code here is a bit complex but what matters is the calls to
% predictPhenotype.m
% NOTE: to replicate this with the most recent version of the toolbox, 
% you'll have to set parameters_prediction.kernel = 'gaussian'; and change 
% the call to cvfolds to not permute (new default is to permute) % CA

% prediction parameters
parameters_prediction = struct();
parameters_prediction.verbose = 0;
parameters_prediction.method = 'KRR';
parameters_prediction.alpha = [0.1 0.5 1.0 5];
parameters_prediction.sigmafact = [1/2 1 2];

for ia = 1:4 % cycle through the structural variables
    
    A = Anatomy{ia}; 
 
    % computing the Euclidean distnces between the structurals 
    D = zeros(size(A,1));
    for n1 = 1:size(A,1)-1
        for n2 = n1+1:size(A,1)
            D(n1,n2) = sqrt( sum( (A(n1,:) - A(n2,:)).^2 ) );
            D(n2,n1) = D(n1,n2);
        end
    end
    
    % performing the prediction
    explained_variance = NaN(size(vars,2),1);
    vars_hat = NaN(N,size(vars,2)); % the predicted variables
    for j = 1:size(vars,2)
        y = vars(:,j); % here probably you need to remove subjects with missing values
        [yhat,stats] = predictPhenotype(y,D,parameters_prediction,twins,conf);
        explained_variance(j) = corr(squeeze(yhat),y).^2;
        vars_hat(:,j) = yhat;
    end

    save('structural_predictions.mat','vars_hat','explained_variance')
        
end


%% Predictions of behaviour using the static FC (with and without structural deconfounding) 
% The static FC is used only through the distance matrices computed previously

Corrected_by_structure = 0; % set this to 0...3 

vars0 = vars; 

switch Corrected_by_structure
    case 1
        vars = vars - vars_hat;
    case 2
        vars = vars - vars_hat;
    case 3
        vars = vars - vars_hat;
end

D = DistStatic; 
explained_variance = NaN(size(vars,2),1);
vars_hat = NaN(N,size(vars,2)); % the predicted variables
for j = 1:size(vars,2)
    y = vars(:,j); % here probably you need to remove subjects with missing values
    [yhat,stats] = predictPhenotype(y,D,parameters_prediction,twins,conf);
    explained_variance(j) = corr(squeeze(yhat),y).^2;
    vars_hat(:,j) = yhat;
end

save('staticFC_predictions.mat','vars_hat','explained_variance')

vars = vars0; 

%% Predictions of behaviour using the HMMs (with and without structural deconfounding) 



Corrected_by_structure = 0; % set this to 0...3 

vars0 = vars; 

switch Corrected_by_structure
    case 1
        vars = vars - vars_hat;
    case 2
        vars = vars - vars_hat;
    case 3
        vars = vars - vars_hat;
end

explained_variance = NaN(size(vars,2),repetitions);
vars_hat = NaN(N,size(vars,2),repetitions); % the predicted variables
for r = 1:repetitions
    D = DistHMM(:,:,r);
    for j = 1:size(vars,2)
        y = vars(:,j); % here probably you need to remove subjects with missing values
        [yhat,stats] = predictPhenotype(y,D,parameters_prediction,twins,conf);
        explained_variance(j,r) = corr(squeeze(yhat),y).^2;
        vars_hat(:,j,r) = yhat;
    end
end

save('HMM_predictions.mat','vars_hat','explained_variance')

vars = vars0; 
