% Code used in Vidaurre et al. (2018) Cerebral Cortex
%
% Detailed documentation and further examples can be found in:
% https://github.com/OHBA-analysis/HMM-MAR
% This pipeline must be adapted to your particular configuration of files.
%%%%%%%%%%%%%%%%%%%%%%%%%
%% Setup
DO_PLOT = 1;

mydir = '/home/diegov/MATLAB/';
addpath(genpath([ mydir 'HMM-MAR']))

load('../Jasper_Tutorial/Data.mat') % load data
% Data is here already prepared (from its original SPM form) to be run by TUDA.
% The Data.mat file contains:
% - data: (time points x channels) data matrix of MEG data for one
%  subject, where trials have been concatenated
% - T: (number of trials x 1) vector containing the length in time points of each trial,
%  such that sum(T) must be equal to the number of time points in 'data'
% - Y: (time points x stimulus features), containing the value of the stimulus to decode,
%  for each time point. This is a grating with a certain orientation;
%  more specifically, this is the difference between the perceived
%  orientation and the orientation that the subject is keeping in memory
%  (see Myers et al, 2015; eLife).
%  Here, the stimulus features are two: the sine and cosine of
%  angle we are to decode, which allows us to treat the angle as if it were
%  Gaussian-distributed.
%
% Note: Y can be also specified as (number of trials x stimulus features)

% It's recommended to standardise the signals at the subject level
% (i.e. for each subject separately),
% so that, for each subject and channel, the variance of the channels is 1.
% This should be done before calling TUDA. Essentially, this is done to
% avoid PCA to give more importance to high-variance channels.
data = zscore(data);

% Options for TUDA
options = struct();
options.K = 4; % number of states / decoders
options.pca = 48; % there are a lot of channels, so we work on PCA space
options.detrend = 1; % detrend the data
options.standardise = 0; % don't standardize the data for each trial
options.parallel_trials = 1; % we assume same experimental paradigm for all trials

% the following are parameters affecting the inference process - not crucial
options.tol = 1e-5;
options.initcyc = 10;
options.initrep = 3;
options.verbose = 1;
options.cyc = 100;

% time of interest; if possible, we prefer to train TUDA using baseline because
% it will try to fit decoders to moments when there are no stimuli, therefore
% fitting to noise and worsening the quality of the decoders.
twin = [0 0.6];
t = linspace(twin(1) , twin(end), T(1)); % time index

%% Run the model
[tuda,Gamma] = tudatrain(data,Y,T,options);

%% Test the model using cross-validation
options_cv = options; options.NCV = 10;
[accuracy,accuracy_star] = tudacv(data,Y,T,options_cv);

%% See if the between-temporal variability in the states predict behaviour
% Here we predict reaction time is predicted by using the state time
% courses. It is assumed that there exists a variable response, telling which
% trials have a button press; and rt, which contains the reaction time itself
keep = response == 1; % which trials the subject pressed the button?
rt = resptime(keep); % reaction time
% State time courses for the trials where there is a button prese
Gamma_keep = reshape(Gamma,[T(1) length(T) options.K]);
Gamma_keep = Gamma_keep(:,keep,:);
Gamma_keep = reshape(Gamma_keep,[T(1)*sum(keep) options.K]);
[rt_hat,ev] = tudaregressbeh(Gamma_keep,T(keep),rt,60);
[rt_hat_trans,ev_trans] = tudaregressbeh(Gamma_keep,T(keep),rt,-1);

save('Outputs.mat','accuracy','accuracy_star','rt_hat','ev',...
    'rt_hat_trans','ev_trans','tuda','Gamma')

%% Do plotting

if DO_PLOT
    
    keep = response == 1; % which trials the subject pressed the button?
    rt = resptime(keep); % reaction time
    
    % Plot the state activation trial by trial and on average
 
    figure(1);clf(1)
    colors = {[0,0,0.8],[0.2,0.7,0.2],[1,0,0],[0.8 0.8 0.2],[0.8 0 0.8]}; % b,g,r,y,m
    
    subplot(212) % Plot the mean activation time of each "decoder"
    MeanGamma = getFractionalOccupancy (Gamma,T,options,1);
    hold on
    for k = 1:options.K
        %h(k).FaceColor =  colors{k};
        plot(t,MeanGamma(:,k),'Color',colors{k},'LineWidth',4)
    end
    hold off
    xlabel('Time (s)'); ylabel('State activation')
    set(gca,'FontSize',16)
    xlim([0 0.45])
    
    subplot(211)
    keep = response == 1; % which trials the subject pressed the button?
    [~,ord_rt] = sort(resptime(keep),'ascend'); % order by reaction time
    GammaCol = zeros(size(Gamma,1),3);
    for k = 1:options.K
        these = sum(repmat(Gamma(:,k),1,options.K-1) > Gamma(:,setdiff(1:options.K,k)),2) == options.K-1;
        GammaCol(these,:) = repmat(colors{k},sum(these),1);
    end
    GammaCol = reshape(GammaCol,[T(1) length(T) 3]); % -> (time by trials by states)
    GammaCol = GammaCol(:,keep,:); % only trials with button press
    GammaCol = permute(GammaCol(:,ord_rt,:),[2 1 3]); % -> (trials by time by states)
    GammaCol = GammaCol(sort(randperm(sum(keep),200)),:,:); % choose some to plot
    imagesc(t,1:size(GammaCol,1),GammaCol)
    xlabel('Time (s)'); ylabel('Trials')
    set(gca,'FontSize',16)
    xlim([0 0.45])
  
    % Plot cross-validated accuracy
    figure(2)
    plot(t,accuracy_star,'LineWidth',3)
    hold on; plot(t,zeros(size(t)),'k'); hold off
    xlabel('Time (s)'); ylabel('Explained variance')
    set(gca,'FontSize',16)
    
    
    % Plot reaction time vs predicted reaction time
    figure(3)
    subplot(121)
    scatter(rt,rt_hat,'filled')
    title('Using raw state time courses')
    xlabel('Reaction Time')
    ylabel('Predicted reaction Time')
    disp(['Correlation ' num2str(corr(rt,rt_hat))])
    set(gca,'FontSize',16)
    subplot(122)
    scatter(rt,rt_hat_trans,'filled')
    title('Transition probability matrix')
    xlabel('Reaction Time')
    ylabel('Predicted reaction Time')
    disp(['Correlation ' num2str(corr(rt,rt_hat_trans))])
    set(gca,'FontSize',16)
    
end