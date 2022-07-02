%% This script contains the analysis presented in Higgins et al 2021:
% "SpatioTemporally Resolved MVPA methods for M/EEG"
% 
% This script is organised in order to be most illustrative of the method;
% it begins by simulating data and fitting the model to demonstrate how the
% model works in simple 2D examples (these simulations are those presented
% in the SI of Higgins et al 2021); it then contains the analysis presented
% in the paper for the two datasets indicated. 
%
% Part 1: Simulations for STRM-Classification model
% Part 2: Simulations for STRM-Regression model
% Part 3: Analysis applied to categorical visual stimulus data
% Part 4: Analysis pipeline for continuous value EEG data
%
% The data for parts 3 and 4 are publically available at Mendeley data:
% Higgins, Cameron (2021), STRM_Datasets, Mendeley Data, V1, 
% doi: 10.17632/jwjkszg4dx.1


%% PART ONE: Simulations for STRM-Classification model:
clear all;
rng(1);
P = 2; % number of data dimensions
Q = 2; % number of classes
K = 2; % number of latent states
T = 50; % number of timepoint in each trial
N = 20; % number of trials

% assign a location to save figures to:
%figdir = ['C:\Users\chiggins\Google Drive\MATLAB\6.0 Classification Work\LDAPaperScripts\STRMSimulationFigures\'];
figdir = '/Users/chiggins/Google Drive/MATLAB/6.0 Classification Work/LDAPaperScripts/STRMSimulationFigures/'
mkdir(figdir);
% generate state timecourses:
z = zeros(N*T,K);
for itrial = 1:N
    transtime = rand(1)*48 + 1;
    t_offset = (itrial-1)*T;
    z(t_offset + [1:floor(transtime)],1)=1;
    z(t_offset + [ceil(transtime):T],2)=1;
end
cols = {[1 0 0],[0 1 0]}
% generate each class activation patterns and covariance matrices:
for ik=1:K
    W{ik} = randn(P+1,Q);
    Sigma{ik} = 0.5*wishrnd(eye(P),P);
end

% alternative: extremely simplified model:
%W{1} = [0,0;1,1;-1,-1];
%W{2} = [0,0;1,-1;-1,1];
%Sigma{1} = 0.25*eye(2);
%Sigma{2} = 0.25*eye(2);

% generate design matrices:
DM = [ones(N*T,1),[ones(N*T/2,1);zeros(N*T/2,1)],[zeros(N*T/2,1);ones(N*T/2,1)]];

% generate data itself:
data = zeros(N*T,P);
for itrial=1:N
    t_offset = (itrial-1)*T;
    for iT = 1:T
        data(t_offset+iT,:) = mvnrnd(DM(t_offset+iT,:)*W{find(z(t_offset+iT,:),1)},Sigma{find(z(t_offset+iT,:),1)});
    end
end
% scatter plot:
figure('Position', [6 478 1885 620]);
subplot(1,3,1);
scatter(data(DM(:,2)==1,1),data(DM(:,2)==1,2),10,cols{1});hold on;
scatter(data(DM(:,3)==1,1),data(DM(:,3)==1,2),10,cols{2});
plot4paper('Channel 1','Channel 2');
title('Scatter plot of all data generated')
legend('Class 1','Class 2')
% plot some data:
t = 1:T;
subplot(2,3,2);
itrial = N/2;t_offset = (itrial-1)*T;
plot(t,data(t_offset + [1:T],:),'LineWidth',2);
title('Sample trial: Class 1');
plot4paper('Time','Channel signal');
z_thistrial = find(z(t_offset + [1:iT],2),1);
hold on;
plot([z_thistrial,z_thistrial],ylim,'k--')
legend({'Channel 1','Channel 2','Transition time'},'Location','EastOutside');
subplot(2,3,5);
itrial = N;t_offset = (itrial-1)*T;
plot(t,data(t_offset + [1:T],:),'LineWidth',2);
title('Sample trial: Class 2');
plot4paper('Time','Channel signal');
z_thistrial = find(z(t_offset + [1:iT],2),1);
hold on;
plot([z_thistrial,z_thistrial],ylim,'k--')

legend({'Channel 1','Channel 2','Transition time'},'Location','EastOutside');
subplot(1,3,3);
gammaRasterPlot(z,T);
plot4paper('Time','Trials');
title('Simulated State Transition times');

print([figdir,'STRMClassification_Fig1'],'-dpng')

%% Fit STRM-Classification model:

options = [];
options.K = K;
options.encodemodel = true; % this activates the STRM model
options.intercept = true;
options.covtype = 'full';
[STRMmodel,STRMstates] = tucatrain(data,DM(:,2:3),repmat(T,N,1),options);

figure('Position',[156 678 1563 420]);
subplot(1,3,1);
gammaRasterPlot(z,T);
plot4paper('Time','Trials');
title('Simulated State Transition times');
subplot(1,3,2);
gammaRasterPlot(STRMstates,T);
plot4paper('Time','Trials');
title('Inferred State Transition Times');

subplot(1,3,3);
vpath = STRMstates==repmat(max(STRMstates,[],2),1,K);
truestates = [z(:,1)-vpath(:,1)==0,z(:,1)-vpath(:,1)~=0];
gammaRasterPlot(truestates,T);
%legend({'Inferred state correct','Inferred state incorrect'})
colormap('Hot')
title('Inferred state == true')
print([figdir,'STRMClassification_Fig2'],'-dpng')

%% Plot STRM model parameters inferred:
figure('Position', [680 180 1020 918]);
for k=1:2
    subplot(2,2,k);
    for iclass=1:2
        scatter(data(vpath(:,k)==1 & DM(:,1+iclass)==1,1),data(vpath(:,k)==1 & DM(:,1+iclass)==1,2),10,cols{iclass})
        hold on;
    end
    title(['Samples assigned to state ',int2str(k)])
    plot4paper('Channel 1','Channel 2')
    legend('Class 1','Class 2');
    xl{k} = xlim();
    yl{k} = ylim();
end

W_exp = tudabeta(STRMmodel);
Om = tudaomega(STRMmodel);
for k=1:2
    subplot(2,2,2+k);
    % add mean to each:
    A = [1,1,0;1,0,1];
    W_exp_k = A*W_exp(:,:,k);
    a = plot(W_exp_k(1,1),W_exp_k(1,2),'+','MarkerSize',20,'MarkerFaceColor',cols{1},'LineWidth',5,'Color',cols{1});
    % a = scatter(W_exp_k(1,1),W_exp_k(1,2),'+','LineWidth',10)
    hold on;
    plot(W_exp_k(2,1),W_exp_k(2,2),'+','MarkerSize',20,'MarkerFaceColor',cols{2},'LineWidth',5,'Color',cols{2})
    xlim(xl{k});ylim(yl{k});
    [XG,YG] = meshgrid(linspace(xl{k}(1),xl{k}(2),100),linspace(yl{k}(1),yl{k}(2),100));
    Sigma_est = inv(Om(:,:,k));
    for iclass=1:2
        for i1=1:100
            for i2=1:100
                ZG(i1,i2) = mvnpdf([XG(1,i1),YG(i2,1)],[W_exp_k(iclass,1),W_exp_k(iclass,2)],Sigma_est);
            end
        end
        contour(XG,YG,ZG','LineWidth',2);hold on;
    end
    a(1) = plot(W_exp_k(1,1),W_exp_k(1,2),'+','MarkerSize',20,'MarkerFaceColor',cols{1},'LineWidth',5,'Color',cols{1});
    % a = scatter(W_exp_k(1,1),W_exp_k(1,2),'+','LineWidth',10)
    hold on;
    a(2) = plot(W_exp_k(2,1),W_exp_k(2,2),'+','MarkerSize',20,'MarkerFaceColor',cols{2},'LineWidth',5,'Color',cols{2})
    legend(a,{'Class 1','Class 2'});
    title(['Model distribution for state ',int2str(k)])
    plot4paper('Channel 1','Channel 2')
end

print([figdir,'STRMClassification_Fig3A'],'-dpng')

%% plot real ground truth distributions:
figure('Position', [680 180 1020 918]);
for k=1:2
    subplot(2,2,2+k);
    % add mean to each:
    A = [1,1,0;1,0,1];
    W_exp_k = A*W{k};
    a = plot(W_exp_k(1,1),W_exp_k(1,2),'+','MarkerSize',20,'MarkerFaceColor',cols{1},'LineWidth',5,'Color',cols{1});
    % a = scatter(W_exp_k(1,1),W_exp_k(1,2),'+','LineWidth',10)
    hold on;
    plot(W_exp_k(2,1),W_exp_k(2,2),'+','MarkerSize',20,'MarkerFaceColor',cols{2},'LineWidth',5,'Color',cols{2})
    xlim(xl{k});ylim(yl{k});
    [XG,YG] = meshgrid(linspace(xl{k}(1),xl{k}(2),100),linspace(yl{k}(1),yl{k}(2),100));
    Sigma_est = Sigma{k};
    for iclass=1:2
        for i1=1:100
            for i2=1:100
                ZG(i1,i2) = mvnpdf([XG(1,i1),YG(i2,1)],[W_exp_k(iclass,1),W_exp_k(iclass,2)],Sigma_est);
            end
        end
        contour(XG,YG,ZG','LineWidth',2);hold on;
    end
    a(1) = plot(W_exp_k(1,1),W_exp_k(1,2),'+','MarkerSize',20,'MarkerFaceColor',cols{1},'LineWidth',5,'Color',cols{1});
    hold on;
    a(2) = plot(W_exp_k(2,1),W_exp_k(2,2),'+','MarkerSize',20,'MarkerFaceColor',cols{2},'LineWidth',5,'Color',cols{2})
    legend(a,{'Class 1','Class 2'});
    title(['Ground truth dist for state ',int2str(k)])
    plot4paper('Channel 1','Channel 2')
end

print([figdir,'STRMClassification_Fig3B'],'-dpng')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% PART 2: Fit STRM-Regression model to simulated data

% this script simulates some data, according to the below relationship:
%
%   data_t = regressors_t * betas_k + epsilon_k
%
% where k denotes the latent state active at that time. Espilon_k are 
% residuals with zero mean but with different covariance matrices in each 
% state.
%
% The script then runs through each of the following steps:
%
% 1. Generate data
% 2. Compare decode accuracy of ridge regression decoding and linear
% gaussian system when using a synchronous setup
% 3. Compare decode accuracy of ridge regression decoding and linear
% gaussian system decoding when using an HMM setup
% 4. Investigate forward model parameters 

%% Step 1: Generate data:
rng(3);
% for now we will simulate data as follows:
ndim = 2;
nTr = 20;
T = 50;
addintercept = true;

T_full = repmat(T,nTr,1);

clear mu Sigma DM;
mu{1} = randn(1,2);%[1,0];
Sigma{1} = 0.5*wishrnd(eye(P),P);%[1,0.95;0.95,1];
mu{2} = randn(1,2);%[0,1];
Sigma{2} = 0.5*wishrnd(eye(P),P);%5*[1,-0.95;-0.95,1];
if addintercept
    offset{1} = randn(1,2);%[1,1];
    offset{2} = randn(1,2);%[1,0];
else
    offset{1} = zeros(1,2);
    offset{2} = zeros(1,2);
end
DM = randn(nTr,1);
% simulate state timecourses:
data = zeros(T,nTr,2);
Gamma_true = zeros(T,nTr,2);

for itr = 1:nTr
    t_stateswitch = floor(rand(1)*48) + 1;%round(0.5*rand(1)*T + T/2);
    Gamma_true(1:floor(t_stateswitch),itr,1) = 1;
    Gamma_true((t_stateswitch+1):end,itr,2) = 1;
    %simulate data in each state:
    data(1:t_stateswitch,itr,:) = repmat(permute(mu{1}*DM(itr,:)+offset{1},[1,3,2]),t_stateswitch,1,1);
    %Y(:,itr,:) = repmat(permute(mu{1}*X(itr,:)+offset{1},[1,3,2]),T,1,1);
    data((t_stateswitch+1):end,itr,:) = repmat(permute(mu{2}*DM(itr,:)+offset{2},[1,3,2]),T-t_stateswitch,1,1);
end
data = reshape(data,[T*nTr,ndim]);
Gamma_true = reshape(Gamma_true,[T*nTr,ndim]);
DM = repelem(DM,T,1);

% finally, add noise:
for t=1:T*nTr
    data(t,:) = data(t,:) + mvnrnd([0;0],Sigma{find(Gamma_true(t,:))});
end


% plot for sanity check - 
figure();
for k=1:size(DM,1)
    col = min([max([0,(DM(k)+1.5) ./3]),1]);
    col = col*[1,0,0.75];
    plot(data(k,1),data(k,2),'*','Color',col);hold on;
end
colmap = [0:0.01:1]'*[1,0,0.75];
colormap(colmap);
colorbar;

plot4paper('Channel 1','Channel 2');
title('Scatter plot of all data generated')
print([figdir,'STRMRegression_Fig1'],'-dpng')

% scatter plot:
figure('Position', [6 478 1885 620]);
subplot(1,3,1);
for k=1:size(DM,1)
    col = (DM(k) - min(DM)) ./ (max(DM)-min(DM));
    col = col*[1,0,0.75];
    plot(data(k,1),data(k,2),'*','Color',col);hold on;
end
colmap = [0:0.01:1]'*[1,0,0.75];
colormap(colmap);
cb = colorbar;
TB = ([-1.5:0.5:1.5]- min(DM)) ./ (max(DM)-min(DM));
set(cb,'Ticks',TB)
set(cb,'TickLabels',[-1.5:0.5:1.5]);
plot4paper('Channel 1','Channel 2');
title('Scatter plot of all data generated')
% plot some data:
t = 1:T;
subplot(2,3,2);
itrial = randi(N/2);t_offset = (itrial-1)*T;
plot(t,data(t_offset + [1:T],:),'LineWidth',2);
ylim([-3,3]);
title(['Sample trial: X=',num2str(DM(t_offset+1),3)]);
plot4paper('Time','Channel signal');
z_thistrial = find(Gamma_true(t_offset + [1:iT],2),1);
hold on;
plot([z_thistrial,z_thistrial],ylim,'k--');
legend({'Channel 1','Channel 2','Transition time'},'Location','EastOutside');
subplot(2,3,5);
itrial = N;t_offset = (itrial-1)*T;
plot(t,data(t_offset + [1:T],:),'LineWidth',2);
ylim([-3,3]);
title(['Sample trial: X=',num2str(DM(t_offset+1),3)]);
plot4paper('Time','Channel signal');
z_thistrial = find(Gamma_true(t_offset + [1:iT],2),1);
hold on;
plot([z_thistrial,z_thistrial],ylim,'k--')
legend({'Channel 1','Channel 2','Transition time'},'Location','EastOutside');

subplot(1,3,3);
gammaRasterPlot(z,T);
plot4paper('Time','Trials');
title('Simulated State Transition times');
print([figdir,'STRMReg_Fig1'],'-dpng')
colormap(parula);
print([figdir,'STRMReg_Fig1a'],'-dpng')

%% Fit STRM-Regression model:

options = [];
options.K = K;
options.encodemodel = true; % this activates the STRM model
options.intercept = true;
options.covtype = 'full';
[STRMmodel,STRMstates] = tucatrain(data,DM,repmat(T,N,1),options);

figure('Position',[156 678 1563 420]);
subplot(1,3,1);
gammaRasterPlot(Gamma_true,T);
plot4paper('Time','Trials');
title('Simulated State Transition times');
subplot(1,3,2);
gammaRasterPlot(STRMstates,T);
plot4paper('Time','Trials');
title('Inferred State Transition Times');

subplot(1,3,3);
vpath = STRMstates==repmat(max(STRMstates,[],2),1,K);
truestates = [Gamma_true(:,1)-vpath(:,1)==0,Gamma_true(:,1)-vpath(:,1)~=0];
gammaRasterPlot(truestates,T);
colormap('Hot')
title('Inferred state == true')
print([figdir,'STRMReg_Fig2'],'-dpng')

%% Plot inferred model parameters:
figure('Position', [680 180 1020 918]);
for k=1:2
    subplot(2,2,k);
    %for iclass=1:2
    %    scatter(Y(vpath(:,k)==1,1),Y(vpath(:,k)==1,2),10,cols{iclass})
    %    hold on;
    %end
    for t=1:size(DM,1)
        col = min([max([0,(DM(t)+1.5) ./3]),1]);
        col = col*[1,0,0.75];
        if vpath(t,k)==1
            plot(data(t,1),data(t,2),'*','Color',col);hold on;
        end
    end
    title(['Samples assigned to state ',int2str(k)])
    plot4paper('Channel 1','Channel 2')
    xl{k} = xlim();
    yl{k} = ylim();
end

W_exp = tudabeta(STRMmodel);
Om = tudaomega(STRMmodel);
for k=1:2
    subplot(2,2,2+k);
    % add mean to second regressor:
    A = [1,0;1,1];
    W_exp_k = A*W_exp(:,:,k);
    xlim(xl{k});ylim(yl{k});
    [XG,YG] = meshgrid(linspace(xl{k}(1),xl{k}(2),100),linspace(yl{k}(1),yl{k}(2),100));
    Sigma_est = inv(Om(:,:,k));
    for iclass=1:2
        for i1=1:100
            for i2=1:100
                ZG(i1,i2) = mvnpdf([XG(1,i1),YG(i2,1)],[W_exp_k(1,:)],Sigma_est);
            end
        end
        contour(XG,YG,ZG','LineWidth',2);hold on;
    end
    h = plot([W_exp_k(:,1)],[W_exp_k(:,2)],'k-','LineWidth',1.5);hold on;
    plot([W_exp_k(2,1)],[W_exp_k(2,2)],'k>','LineWidth',1.5);
    hold on;
    
    title(['Model distribution for state ',int2str(k)])
    plot4paper('Channel 1','Channel 2')
end
print([figdir,'STRMReg_Fig3A'],'-dpng')

%% and plot ground truth distribution:
figure('Position', [680 180 1020 918]);
for k=1:2
    subplot(2,2,2+k);
    % add mean to each:
    A = [1,0;1,1];
    W_exp_k = A*[offset{k};mu{k}];
    a = plot(W_exp_k(1,1),W_exp_k(1,2),'+','MarkerSize',20,'MarkerFaceColor',cols{1},'LineWidth',5,'Color',cols{1});
    xlim(xl{k});ylim(yl{k});
    [XG,YG] = meshgrid(linspace(xl{k}(1),xl{k}(2),100),linspace(yl{k}(1),yl{k}(2),100));
    Sigma_est = inv(Om(:,:,k));
    for iclass=1:2
        for i1=1:100
            for i2=1:100
                ZG(i1,i2) = mvnpdf([XG(1,i1),YG(i2,1)],[W_exp_k(1,:)],Sigma_est);
            end
        end
        contour(XG,YG,ZG','LineWidth',2);hold on;
    end
    h = plot([W_exp_k(:,1)],[W_exp_k(:,2)],'k-','LineWidth',1.5);hold on;
    plot([W_exp_k(2,1)],[W_exp_k(2,2)],'k>','LineWidth',1.5);
    hold on;
    title(['Ground truth dist for state ',int2str(k)])
    plot4paper('Channel 1','Channel 2')
end

print([figdir,'STRMReg_Fig3B'],'-dpng')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%% PART THREE: Visual MEG stimuli

fulldir = '/Users/chiggins/data/STRMData/MEGData/'; % project directory where data is stored
datafile = @(sjnum) [fulldir,'FLISj',int2str(sjnum),'.mat'];


%% Step 1: Full CV analysis estimating accuracies:

K_options = [4:2:22]; % Range of parameter values for K to test
Sjs_to_do = [1:22];
% note that for accuracy tests, we crossvalidate; we do not save model
% parameters, only the accuracies; for interrogation of model accuracies we
% refit the models later

% setup save destination
accfile = @(sjnum) [fulldir,'AccSj',int2str(sjnum),'.mat'];

%set static options:
options_classifier = [];
options_classifier.NCV = 10;
options_classifier.CVmethod = [1 2]; %test different options for fitting methods in cross validated test set
options_classifier.cyc = 1;
options_classifier.classifier = 'LDA';
options_classifier.covtype = 'full';

for iSj = Sjs_to_do
    acc_LDA_HMM = zeros(50,length(options_classifier.CVmethod),length(K_options));
    AUC_LDA_HMM = zeros(50,length(options_classifier.CVmethod),length(K_options));
    K_acc_LDA_HMM = zeros(max(K_options),length(options_classifier.CVmethod),length(K_options));
    
    % load data:
    fname = datafile(iSj);
    clear data options_preproc 
    load(fname,'data','options_preproc');
    
    % and decode using LDA classifiers:
    data.X_train = normalise(data.X_train);
    rng(1); %ensure cross validation consistent over multiple runs
    [acc_LDA_synchronous,~,~,AUC_LDA_synchronous] = standard_classification(data.X_train,data.Y_train,data.T_train,options_classifier);
    
    % now fit STRM model:
    for iK = 1:length(K_options)
        K = K_options(iK);
        options_classifier.K=K;
        rng(1); %ensure cross validation consistent
        [~,acc1,~,~,~,acc3,AUC] = tucacv(data.X_train,data.Y_train,data.T_train,options_classifier);
        acc_LDA_HMM(:,:,iK) = squeeze(acc1);
        K_acc_LDA_HMM(1:K,:,iK) = acc3;
        AUC_LDA_HMM(:,:,iK) = AUC;
    end
    fnameout = accfile(iSj);
    save(fnameout,'acc_LDA_HMM','K_acc_LDA_HMM','AUC_LDA_HMM',...
        'options_classifier','K_options',...
        'acc_LDA_synchronous','AUC_LDA_synchronous');
end

%% compare to some alternative (synchronous) classifiers for comparison:

% setup save destination
accsynchfile = @(sjnum) [fulldir,'Acc_synch_Sj',int2str(sjnum),'.mat'];

%set static options:
options_classifier = [];
options_classifier.NCV = 10;

synch_class_labels = {'Naive Bayes','Linear SVM','RBF SVM','Logistic Regression - binomial','Logistic Regression - multinomial','KNN'};
for iSj = Sjs_to_do
    
    fprintf(['\n\n Subject ',int2str(iSj),'\n\n']);
    
    fname = datafile(iSj);
    clear data options_preproc
    load(fname,'data','options_preproc');
    data.X_train = normalise(data.X_train);
    
    % and decode using different classifiers:
    % will test here SVM; NB; LogisticRegression (one vs all); logreg
    % (multinomial)
    rng(1); %ensure cross validation consistent
    options_classifier.classifier = 'LDA';
    options_classifier.covtype = 'diag'; % this corresponds to a Naive Bayes classifier
    [acc1,~,~,AUC,acc2] = standard_classification(data.X_train,data.Y_train,data.T_train,options_classifier);
    acc_class_synchronous(:,1) = acc1;
    AUC_class_synchronous(:,1) = AUC;

    rng(1); %ensure cross validation consistent
    options_classifier.classifier = 'SVM';
    [acc1,~,~,AUC,acc2] = standard_classification(data.X_train,data.Y_train,data.T_train,options_classifier);
    acc_class_synchronous(:,2) = acc1;
    AUC_class_synchronous(:,2) = AUC;

    rng(1); %ensure cross validation consistent
    options_classifier.classifier = 'SVM_rbf';
    [acc1,~,~,AUC,acc2] = standard_classification(data.X_train,data.Y_train,data.T_train,options_classifier);
    acc_class_synchronous(:,3) = acc1;
    AUC_class_synchronous(:,3) = AUC;

    rng(1); %ensure cross validation consistent
    options_classifier.classifier = 'logistic';
    options_classifier.regtype = 'L2';
    [acc1,~,~,AUC,acc2] = standard_classification(data.X_train,data.Y_train,data.T_train,options_classifier);
    acc_class_synchronous(:,4) = acc1;
    AUC_class_synchronous(:,4) = AUC;

    % now do multinomial log reg:
    rng(1);
    Ytemp = data.Y_train;
    [Ytemp,~] = find(Ytemp');
    [acc1,~,~,AUC,acc2] = standard_classification(data.X_train,Ytemp,data.T_train,options_classifier);
    acc_class_synchronous(:,5) = acc1;
    AUC_class_synchronous(:,5) = AUC;


    rng(1); %ensure cross validation consistent
    opts = options_classifier;
    opts.classifier = 'KNN';
    for ik=1:20
        opts.K = (2*ik)-1;
        synch_class_labels{5+ik} = ['KNN:K=',int2str(opts.K)];
        [acc1,~,~,AUC,acc2] = standard_classification(data.X_train,data.Y_train,data.T_train,opts);
        acc_class_synchronous(:,5+ik) = acc1;
        AUC_class_synchronous(:,5+ik) = AUC;
    end
    
    fnameout = accsynchfile(iSj);
    save(fnameout,'acc_class_synchronous','AUC_class_synchronous','synch_class_labels');
end

%% compare to alternatives that don't allow states to move:

% note that for accuracy tests, we crossvalidate; we do not save model
% parameters, only the accuracies; for interrogation of model accuracies we
% refit the models later

% setup save destination
accfile = @(sjnum) [fulldir,'AccSj',int2str(sjnum),'.mat'];

%set static options:
options_classifier = [];
options_classifier.NCV = 10;
options_classifier.CVmethod = [1]; %use mean fit only
options_classifier.classifier = 'LDA';
options_classifier.covtype = 'full';
options_classifier.cyc = 1;

windows = 1:10;

for iSj = Sjs_to_do
    acc_LDA_slidingwindow = zeros(50,length(windows));
    AUC_LDA_slidingwindow = zeros(50,length(windows));
    
    fnameout = accfile(iSj);
    
    % load data:
    fname = datafile(iSj);
    clear data options_preproc 
    load(fname,'data','options_preproc');
    
    % and decode using LDA classifiers:
    data.X_train = normalise(data.X_train);
        
    rng(1); %ensure cross validation consistent

    % apply sliding window averaging
    for iwindow=1:length(windows)
        options_classifier.slidingwindow = windows(iwindow);
        [acc1,~,~,AUC] = standard_classification(data.X_train,data.Y_train,data.T_train,options_classifier);
        acc_LDA_slidingwindow(:,iwindow) = acc1;
        AUC_LDA_slidingwindow(:,iwindow) = AUC;
    end
          
    fnameout = accfile(iSj);
    save(fnameout,'acc_LDA_slidingwindow','AUC_LDA_slidingwindow','-append');
end

%% and fit non cross-validated models:

K_options = 8; % set to single value

% setup save destination:
modelfile = @(sjnum) [fulldir 'ModelSj',int2str(sjnum),'.mat'];

%set static options:
options_classifier = [];
options_classifier.classifier = 'LDA';
options_classifier.covtype = 'full';


%% train models:

for iSj = Sjs_to_do
    fname = datafile(iSj);
    clear data options_preproc model_LDA_HMM 
    load(fname,'data','options_preproc');
    data.X_train = normalise(data.X_train);
    
    model_LDA_HMM = cell(length(K_options),1);
    Gamma_LDA = cell(length(K_options),1);
    for iK = 1:length(K_options)
        K = K_options(iK);
        options_classifier.K=K;
        rng(1); %ensure cross validation consistent
        [model_temp,Gamma_LDA{iK}] = tucatrain(data.X_train,data.Y_train,data.T_train,options_classifier);
        model_LDA_HMM{iK}.beta = tudabeta(model_temp);
        model_LDA_HMM{iK}.Sigma= tudaomega(model_temp);
    end
    fnameout = modelfile(iSj);
    save(fnameout,'model_LDA_HMM','Gamma_LDA');
end

%% Figure 1: Cross-validated Accuracy plots:

figdir = [fulldir,'Figure1/'];
if ~isdir(figdir)
    mkdir(figdir);
end

accfile = @(sjnum) [fulldir,'AccSj',int2str(sjnum),'.mat'];
accsynchfile = @(sjnum) [fulldir,'Acc_synch_Sj',int2str(sjnum),'.mat'];


% iterate through subjects, loading all accuracy data:
acc_hmm_all = zeros(50,2,length(K_options),length(Sjs_to_do));
acc_synch_all = nan(50,length(Sjs_to_do));
for iSj = Sjs_to_do
    fname = accfile(iSj);
    load(fname)
    acc_hmm_all(:,:,:,iSj) = acc_LDA_HMM;
    acc_synch_all(:,iSj) = acc_LDA_synchronous;
end
acc_hmm_all = acc_hmm_all(:,:,:,Sjs_to_do);
acc_synch_all = acc_synch_all(:,Sjs_to_do);


% set global parameters:
ifitmethod = 2; % this for the regression fit CV estimates outlined in HBM paper
acc_hmm = squeeze(acc_hmm_all(:,ifitmethod,:,:));

% cross validate timepoint by timepoint to determine K:
for iSj=Sjs_to_do
    SJ_in = setdiff(Sjs_to_do,iSj);
    for it=1:50
        [~,K_best_t_CV(iSj,it)] = max(mean(mean(acc_hmm(it,:,SJ_in),1),3));
        acc_hmm_CV_t(it,iSj) = squeeze(acc_hmm(it,K_best_t_CV(iSj,it),iSj));
    end
end

figure('Position',[440 378 936 420]);
m = mean(acc_hmm_CV_t,2);
s = std(acc_hmm_CV_t,[],2);
t = 0.01:0.01:0.5;
shadedErrorBar(t,m,s./sqrt(nSj),{'Color','Blue','LineWidth',2},1); hold on;
m = mean(acc_synch_all,2);
s = std(acc_synch_all,[],2);
shadedErrorBar(t,m,s./sqrt(nSj),{'Color','Red','LineWidth',2},1); hold on;
plot4paper('Time (sec)','Accuracy');
line(t,0.125*ones(length(t),1),'Color','Black','LineWidth',1,'LineStyle','--');


% and find significance:
thresh = 1;
[corrp,tstats] = osl_clustertf(permute(acc_hmm_CV_t-acc_synch_all,[2,3,1]),thresh);
corrp(corrp<0.95) = NaN;
corrp(corrp>0.95) = 1;
plot(t,0.1*corrp,'LineWidth',2,'Color',[0.5,0.5,0.5]);
YL = ylim;
ylim([0.08,YL(2)]);
h(1) = plot([NaN,NaN],[1 1],'Color','Blue','LineWidth',2);
h(2) = plot([NaN,NaN],[1 1],'Color','Red','LineWidth',2);
legend(h,{'STRM-Classifier','Timepoint-by-timepoint LDA'},'Location','EastOutside')

print([figdir,'Acc_HMM_CV_t'],'-dpng');

% and compare with sliding window:
for iSj = Sjs_to_do
    fnameout = accfile(iSj);
    load(fnameout,'acc_LDA_slidingwindow','AUC_LDA_slidingwindow');
    acc_SW(:,:,iSj) = acc_LDA_slidingwindow;
    AUC_SW(:,:,iSj) = AUC_LDA_slidingwindow;
end

% cross validate to optimise window size:
for iSj=Sjs_to_do
    SJ_in = setdiff(Sjs_to_do,iSj);
    for it=1:50
        [~,W_best_t_CV(iSj,it)] = max(mean(acc_SW(it,:,SJ_in),3),[],2);
        acc_SW_CV(it,iSj) = acc_SW(it,W_best_t_CV(iSj,it),iSj);
    end
end

figure('Position',[440 378 936 420]);
m = mean(acc_hmm_CV_t,2);
s = std(acc_hmm_CV_t,[],2);
t = 0.01:0.01:0.5;
shadedErrorBar(t,m,s./sqrt(nSj),{'Color','Blue','LineWidth',2},1); hold on;
m = mean(acc_synch_all,2);
s = std(acc_synch_all,[],2);
shadedErrorBar(t,m,s./sqrt(nSj),{'Color','Red','LineWidth',2},1); hold on;
plot4paper('Time (sec)','Accuracy');
line(t,0.125*ones(length(t),1),'Color','Black','LineWidth',1,'LineStyle','--');
m = mean(acc_SW_CV,2);
s = std(acc_SW_CV,[],2);
t = 0.01:0.01:0.5;
shadedErrorBar(t,m,s./sqrt(nSj),{'Color','Green','LineWidth',2},1); hold on;

legend('STRM-Classifier','Timepoint-by-timepoint LDA','Sliding Window LDA');

print([figdir,'Acc_HMM_vsSW_CV_t'],'-dpng');

% and plot accuracy vs other ML classifiers

clear acc AUC;
for iSj=Sjs_to_do
    
    filenamein = accsynchfile(iSj);
    load(filenamein);
    acc(:,:,iSj) = squeeze(acc_class_synchronous);
    AUC(:,:,iSj) = squeeze(AUC_class_synchronous);
end
acc_synch = acc(:,1:5,:);
auc_synch = AUC(:,1:5,:);

% cross validate to choose best value of K for KNN:
clear acc_KNN
for iSjHO=1:nSj
    SjIn = setdiff(1:nSj,iSjHO);
    m = mean(acc(:,6:end,SjIn),3);
    for t=1:50
        [~,KNN_KCV(iSjHO,t)] = max(m(t,:));
        acc_KNN(t,1,iSjHO) = acc(t,5+KNN_KCV(iSjHO,t),iSjHO);
        auc_KNN(t,1,iSjHO) = AUC(t,5+KNN_KCV(iSjHO,t),iSjHO);
    end
end
acc_synch = cat(2,acc_synch,acc_KNN);
auc_synch = cat(2,auc_synch,auc_KNN);

acc_synch = cat(2,permute(acc_synch_all,[1,3,2]),acc_synch);
acc_synch = cat(2,permute(acc_hmm_CV_t,[1,3,2]),acc_synch);
acc_all_t = squeeze(mean(acc_synch,1));

figure();
errorbar([1:8],mean(acc_all_t'),std(acc_all_t')./sqrt(nSj),'*','LineWidth',2)

xlim([0.5,8.5]);
set(gca,'XTick',[1:8]);
synch_class_labels{4} = 'LR: binomial';
synch_class_labels{5} = 'LR: multinomial';
set(gca,'XTickLabel',{'STRM-Classifier','Timepoint-by-timepoint LDA',synch_class_labels{1:5},'KNN'});
set(gca,'XTickLabelRotation',45);
plot4paper('Classifier','Accuracy');
yl = ylim();
clear paireddiff;
for i=1:7
    [~,pvals_HMM(i)] = ttest(acc_all_t(1,:)-acc_all_t(1+i,:));hold on;
    if pvals_HMM(i)<1e-5
        plot(i+1,yl(2)-0.05*(diff(yl)),'k*','LineWidth',2);
    end
    paireddiff(:,i) = acc_all_t(1,:)-acc_all_t(1+i,:);
end

line([1.5,1.5],[0,0.5],'Color', 'black');
print([figdir,'Acc_HMM_vs_MLClassifiers'],'-dpng');

%% Figure 2: Activity maps:

figdir = [fulldir,'Figure2/'];
mkdir(figdir);

% timecourse plots:
K = 8;
iK=1; % we only have one value of K to select
Gamma_all = [];RTs = [];T_Sj_all = [];
for iSj = Sjs_to_do
    fnamein = modelfile(iSj);
    load(fnamein)
    fnamein = datafile(iSj);
    load(fnamein)
    Gamma_all = [Gamma_all;Gamma_LDA{iK}];
    T_Sj_all = [T_Sj_all,length(Gamma_LDA{iK})];
    RTs = [RTs;normalise(data.reactionTimes_train)];
    %subplot(5,5,iSj);
    figure( 'Position', [440 56 248 749]);
    gamtemp = reshape(Gamma_LDA{iK},50,size(Gamma_LDA{iK},1)/50,K);
    
    gammean(:,:,iSj) = mean(gamtemp,2);
    [~,inds] = sort(data.reactionTimes_train);
    gamtemp = gamtemp(:,inds,:);
    gamtemp = reshape(gamtemp,length(inds)*50,K);
    gammaRasterPlot(gamtemp,50);
    set(gcf, 'InvertHardCopy', 'off');
    print([figdir,'SJ',int2str(iSj),'RasterPlots_RTsorted'],'-depsc');
end
close all;

figure( 'Position', [440 56 248 749]);
gamtemp = reshape(Gamma_all,50,size(Gamma_all,1)/50,K);
[RTs,inds] = sort(RTs);
gamtemp = gamtemp(:,inds,:);
gamtemp = reshape(gamtemp,length(inds)*50,K);
gammaRasterPlot(Gamma_all,50);
set(gcf, 'InvertHardCopy', 'off');
print([figdir,'AllSJRasterPlot'],'-depsc');

figure('Position', [187 600 1212 205]);
cols = parula(8);
t=0.01:0.01:0.5;
clear h;
for k=1:8
    shadedErrorBar(t,mean(gammean(:,k,:),3),std(gammean(:,k,:),[],3),{'Color',cols(k,:),'LineWidth',2});
    hold on;
    leglabels{k} = ['State ',int2str(k)];
    h(k) = plot(NaN,NaN,'Color',cols(k,:),'LineWidth',2);
end
legend(h,leglabels,'Location','EastOutside');
ylim([0,1]);
plot4paper('Time','State Probability');
set(gcf, 'InvertHardCopy', 'off');
print([figdir,'AllSJ_meanGamma'],'-depsc');

% and maps themselves:

globalthresh = true;
sourcedatafile = @(sjnum) [fulldir 'souredataSj',int2str(sjnum),'.mat'];

clear Betas Sigmas M;
for iSj = Sjs_to_do
    fnamein = modelfile(iSj);
    load(fnamein)
    fname = datafile(iSj);
    load(fname,'data');
    fnamesource = sourcedatafile(iSj);
    load(fnamesource,'X_source');
    data = [ones(length(data.Y_train),1),data.Y_train];
    Y2=[];
    for ik=1:K_options(iK)
        Y2 = [Y2,data.*repmat(Gamma_LDA{iK}(:,ik),[1,size(data,2)])];
    end
    
    temp = pinv(Y2)*X_source;
    resid = X_source - Y2*temp;
    for ik=1:K
        K_set = [1:size(data,2)] + (ik-1)*(size(data,2));
        Betas(:,:,ik,iSj) = temp(K_set,:);
        Sigmas(:,ik,iSj) = var(resid,Gamma_LDA{iK}(:,ik));
    end
end

parc_file = ['fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm'];
parc = parcellation(parc_file);
% compute f statistic per subject:
for ik=1:K
    for iSj=Sjs_to_do
        for ich = 1:size(Betas,2)
            % sompute variance between group means:
            numerator = var(Betas(2:end,ich,ik,iSj),[],1);
            denom = Sigmas(ich,ik,iSj);
            f_stat(ich,ik,iSj) = numerator./denom;
        end
    end
end

f_stat_group = mean(f_stat,3);
if globalthresh
    thresh = prctile(f_stat_group(:),75);
end
CA = [0, max(f_stat_group(:))];
for ik = 1:K
    if ~globalthresh
        thresh = prctile(f_stat_group(:,ik),75);
    end
    dat = f_stat_group(:,ik);dat(dat<thresh)=NaN;
    plot_surf_summary_neuron(parc,dat,1,[],[],[],[],CA);
    print([figdir,'activitymap_K',int2str(K),'_Fstat_St',int2str(ik)],'-depsc');
    set(gcf, 'InvertHardCopy', 'off');
    close(gcf);
end

%% figure 3: Timecourse modulation

figdir = [fulldir,'Figure3/'];
mkdir(figdir);

prestimdatafile = @(sjnum) [fulldir,'prestimdata/PSDSj',int2str(sjnum),'.mat'];

n_modes = 2;

% fit nnmf to prestimulus power to find 2 main modes of variation:
nnmf_file = [fulldir,'nnmf_file',int2str(n_modes),'.mat']
if ~isfile(nnmf_file)
    PX = [];SjInds=[];
    for iSj=Sjs_to_do
        load(prestimdatafile(iSj),'pxx','f');
        
        % remove outliers:
        s = sum(sum(pxx,1),3);
        badtrls{iSj} = find(abs(s-mean(s))>2*std(s));
        pxx(:,badtrls{iSj},:) = [];
        
        PX = cat(2,PX,pxx);
        
        Sjindstemp = zeros(size(pxx,2),length(Sjs_to_do));
        Sjindstemp(:,iSj) = 1;
        SjInds = [SjInds;Sjindstemp];
    end
    nch = size(PX,3);
    lowestresid = -1;
    PX = permute(PX,[2,1,3]);
    f_in = find(f>1.5);% take power in bands from 1.5Hz up
    PX = PX(:,f_in,:); 
    f = f(f_in);
    for isamp=1:10 % run ten times and take best fit
        [A_temp,B_temp] = nnmf(PX(:,:),n_modes);
        PX_est = A_temp*B_temp;
        resid = mean((PX_est(:)-PX(:)).^2);
        if resid<lowestresid || lowestresid==-1
            lowestresid = resid;
            A_all = A_temp;
            B_all = B_temp;
            B_toplot = pinv(A_temp)*PX(:,:);
        end
    end
    B_all = reshape(B_all,[n_modes,size(f),nch]);
    save(nnmf_file,'A_all','B_all','n_modes','SjInds');
else
    load(nnmf_file);
    load(psdfile(1),'f');
end
%%
clear B_gmm;
C = zeros(2+n_modes);

for iSj=Sjs_to_do
    fnamein = modelfile(iSj);
    load(fnamein)

    gamtemp = reshape(Gamma_LDA{iK},[50,size(Gamma_LDA{iK},1)/50,size(Gamma_LDA{iK},2)]);
    gamtemp = permute(gamtemp,[2,1,3]);
    
    % compute transition times T:
    [~,vpath] = max(Gamma_LDA{iK},[],2);
    vpath = reshape(vpath,[50,size(Gamma_LDA{iK},1)/50]);
    % remove outliers:
    gamtemp(badtrls{iSj},:,:) = [];
    vpath(:,badtrls{iSj}) = [];
    
    clear Ttrans
    for i=1:K_options(iK)-1
        for itr = 1:size(vpath,2)
            Ttrans(itr,i) = find([vpath(:,itr);K+1]>i,1);
        end
    end
    fname = datafile(iSj);
    load(fname,'data');
    
    Ttrans = demean(Ttrans,1) ./ 100;
    Ttrans(isnan(Ttrans)) = 0; % where state timecourses don't vary
    % remove outliers:
    TTtemp = normalise(Ttrans);
    TTtemp(isnan(TTtemp))=0;
    badtrls_tt{iSj} = find(any(abs(TTtemp)>3,2));

    % setup subject specific design matrix
    DM = zeros(size(Ttrans,1),2+n_modes);
    DM1 = data.reactionTimes_train;
    DM2 = data.ISI_train;
    DM1(badtrls{iSj}) = [];
    DM2(badtrls{iSj}) = [];
    
    DM(:,1) = DM1;
    DM(:,2) = DM2;
    DM(:,3:end) = (A_all(SjInds(:,iSj)==1,:));
    DM(badtrls_tt{iSj},:) = [];
    Ttrans(badtrls_tt{iSj},:) = [];
    DM = normalise(DM);
    DM = decorrelateDesignMatrix(DM);
    DM(:,end+1) = ones(length(DM),1);
    
    % fit GLM:
    B_gmm(:,:,iSj) = pinv(DM)*Ttrans;
    
end

% t-tests with bonferroni correction:
DMlabels = {'Reaction Times','ISI times'};
for n=1:n_modes
    DMlabels{2+n} = ['Mode ',int2str(n)];
end
for k=1:7
    xlabels{k} = [int2str(k), ' to ',int2str(k+1)];
end
for i1=1:2
figure('Position',[138 227 498 565]);
for k=1:size(B_gmm,1)-1
    subplot(ceil(0.5*(size(B_gmm,1)-1)),2,k);
    B_mean = mean(B_gmm(k,:,:),3);
    bar(B_mean);hold on;
    m = max(abs(squash(mean(B_gmm,3))));
    for i=1:size(B_gmm,2)
        [~,pvals(k,i)] = ttest(B_gmm(k,i,:));
        if pvals(k,i) < 0.05/((size(B_gmm,1)-1)*(size(B_gmm,2)))
            plot(i-0.2,1.1*m,'k*','MarkerSize',15,'LineWidth',2);
            plot(i+0.2,1.1*m,'k*','MarkerSize',15,'LineWidth',2);
        elseif pvals(k,i)<0.05 && i1==1
            plot(i,1.1*m,'k*','MarkerSize',15,'LineWidth',1);
        end
    end

    ylim([-1.2,1.2]*m);
    plot4paper('State Transition','Timing modulation (sec)');
    title(DMlabels{k});
    set(gca,'XTick',[1:size(B_gmm,2)]);
    set(gca,'XTickLabel',xlabels);
    set(gca,'XTickLabelRotation',45);
    set(gca,'YTickLabel',get(gca,'YTick'))

end
if i1==1
    print([figdir 'TransitionTimePredictiveModelling'],'-depsc')
else
    print([figdir 'TransitionTimePredictiveModelling_bonfonly'],'-depsc')
end
end

% and plot the modes over frequency:
figure('Position', [1 559 778 239]);
B_toplot = reshape(B_toplot,size(B_all));
for i=1:n_modes
    subplot(1,n_modes,i);
    plot(f,squeeze(B_toplot(i,:,:)),'LineWidth',1.5);
    plot4paper('Frequency','Power');
    YT = get(gca,'YTick');
    YT = [YT(1),YT(end)/2,YT(end)];
    set(gca,'YTick',YT);
end
print([figdir 'TransitionTimePredictiveModelling_',int2str(n_modes),'modes_spectra'],'-depsc')

% and space:
for i=1:n_modes
    toplot = squeeze(sum(B_all(i,:,:),2));
    plot_surf_summary_neuron(parc,toplot,1,[],[],[],[],[]);
    print([figdir 'NNMF_',int2str(n_modes),'modes_M',int2str(i)],'-depsc')
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% PART 4: EEG Data analysis
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
data_dir_foraging = '/Users/chiggins/data/STRMData/EEGData/';

%% Evaluate STRM-Regression Cross-validated Accuracy

%%%% Fit a range of values of K:
K_options = [4:2:22]; 

%set static options:
options_LGS = [];
options_LGS.NCV = 10;
options_LGS.CVmethod = [1,2]; %test all options here
options_LGS.intercept = 1;   
options_LGS.encodemodel = true;
options_LGS.cyc = 1;
options_LGS.covtype = 'full';
options_LGS.accuracyType = 'Pearson';

accfile = @(sjnum) [data_dir_foraging,'AccSj',int2str(sjnum),'.mat'];
datafile = @(sjnum) [data_dir_foraging,'FLISj',int2str(sjnum),'.mat'];

Sjs_to_do = [1:46]; % note there are 23 subjects, 2 sessions each
nregressors = 1;

% fit the classifiers:
for iSj = Sjs_to_do
    
    acc_LGS_HMM = zeros(100,nregressors,length(options_LGS.CVmethod),length(K_options));
    K_acc_LGS_HMM = zeros(length(K_options),nregressors,length(options_LGS.CVmethod),length(K_options));
    
    fname = datafile(iSj);
    clear data DM T varexpl
    load(fname,'data','DM','T');
    
    data = DM(:,1); % this is the stimulus value
    
    % and decode using LDA classifiers:
    rng(1); %ensure cross validation consistent
    [acc_LGS_synchronous] = standard_decoding(data,data,T,options_LGS);

    for iK = 1:length(K_options)
        K = K_options(iK);
        options_LGS.K=K;
        rng(1); %ensure cross validation consistent
        [~,acc_LGS_HMM(:,:,:,iK,:),~,~,~,K_acc_LGS_HMM(1:K_options(iK),:,:,iK,:)] = tudacv(data,data,T,options_LGS);
        % note second dimension is different regressors; third is
        % different state fitting methods
    end
    
    fnameout = accfile(iSj);
    save(fnameout,'acc_LGS_synchronous','acc_LGS_HMM','K_acc_LGS_HMM','options_LGS','K_options');
end


%% and comparison with non-LGS methods for prediction:

% ridge regression decoding
% SVM regression

%set static options:
options_reg = [];
options_reg.NCV = 10;
options_reg.accuracyType = options_LGS.accuracyType;

% setup save destination
accsynchfile = @(sjnum) [data_dir_foraging,'Acc_synch_Sj',int2str(sjnum),'.mat'];
datafile = @(sjnum) [data_dir_foraging,'FLISj',int2str(sjnum),'.mat'];

synch_class_labels = {'Linear Regression','Linear Regression w intercept','Linear SVM','RBF SVM'};

for iSj = Sjs_to_do
    fprintf(['Decoding Subject ',int2str(iSj),'\n']);
    acc_LGS_synchronous = zeros(100,3,1);
    
    fname = datafile(iSj);
    clear data DM T
    load(fname,'data','DM','T');
    
    data = DM(:,1);
    
    rng(1); %ensure cross validation consistent
    
    options_reg.intercept = 1;
    acc_LGS_synchronous(:,1) = standard_decoding(data,data,T,options_reg);

    options_regSVM = options_reg;
    options_regSVM.SVMreg = true;
    options_regSVM.SVMkernel = 'linear';
    acc_LGS_synchronous(:,2) =  standard_decoding(data,data,T,options_regSVM);
    
    options_regSVM.SVMkernel = 'rbf';
    acc_LGS_synchronous(:,3) = standard_decoding(data,data,T,options_regSVM);
    
    fnameout = accsynchfile(iSj);
    save(fnameout,'acc_LGS_synchronous','synch_class_labels');
end

%% compare to equivalent sliding window methods:

options_LGS = [];
options_LGS.NCV = 10;
options_LGS.intercept = 1;   
options_LGS.encodemodel = true;
options_LGS.covtype = 'full';
options_LGS.accuracyType = 'Pearson';


windows = 1:10;
accfile_SW = @(sjnum) [data_dir_foraging,'Acc_SW_Sj',int2str(sjnum),'.mat'];

for iSj = Sjs_to_do
    acc_LGS_slidingwindow = zeros(100,length(windows));
    
    fprintf(['Decoding Subject ',int2str(iSj),'\n']);
    
    fname = datafile(iSj);
    clear data DM T varexpl
    load(fname,'data','DM','T');
    
    data = DM(:,1);
    
    rng(1); %ensure cross validation consistent
    
    % apply sliding window averaging
    for iwindow=1:length(windows)
        options_LGS.slidingwindow = windows(iwindow);
        acc_LGS_slidingwindow(:,iwindow,:) = squeeze(standard_decoding(data,data,T,options_LGS));
    end
    
    fnameout = accfile_SW(iSj);
    save(fnameout,'acc_LGS_slidingwindow');
end


%% Finally, fit non cross-validated models:

K_options = 8;

% setup save destination
datafile = @(sjnum) [data_dir_foraging 'FLISj',int2str(sjnum),'.mat'];
modelfile = @(sjnum) [data_dir_foraging 'ModelSj',int2str(sjnum),'.mat'];

%set static options:
options_LGS = [];
options_LGS.intercept = 1;   
options_LGS.encodemodel = true;
options_LGS.covtype = 'full';
options_LGS.cyc = 1;
options_LGS.useParallel = false;

% train models:
for iSj = Sjs_to_do
    fname = datafile(iSj);
    clear data DM T
    load(fname,'data','DM','T');
    data = DM(:,1); % note Y is just stimulus value
    
    model_LGS_HMM = cell(length(K_options),1);
    for iK = 1:length(K_options)
        K = K_options(iK);
        options_LGS.K=K;
        options_LGS.verbose=false;
        rng(1); %ensure cross validation consistent
        [model_temp,Gamma_LGS{iK}] = tudatrain(data,data,T,options_LGS);
        model_LGS_HMM{iK}.beta = tudabeta(model_temp);
        model_LGS_HMM{iK}.Sigma= tudaomega(model_temp);
    end
    fnameout = modelfile(iSj);
    save(fnameout,'model_LGS_HMM','Gamma_LGS','K_options','options_LGS');
end

%% Figure 1: Cross validated accuracy plots:

figdir = [data_dir_foraging,'Figure1/'];
if ~isdir(figdir)
    mkdir(figdir);
end

accfile = @(sjnum) [data_dir_foraging,'AccSj',int2str(sjnum),'.mat'];
accsynchfile = @(sjnum) [data_dir_foraging,'Acc_synch_Sj',int2str(sjnum),'.mat'];

% iterate through subjects, loading all accuracy data:
load(accfile(1));
acc_hmm_all = zeros(100,2,length(K_options),length(Sjs_to_do));
acc_synch_all = nan(100,length(Sjs_to_do));
for iSj = Sjs_to_do
    fname = accfile(iSj);
    load(fname)
    acc_hmm_all(:,:,:,iSj) = acc_LGS_HMM;
    acc_synch_all(:,iSj) = acc_LGS_synchronous;
end
acc_hmm_all = acc_hmm_all(:,:,:,Sjs_to_do);
acc_synch_all = acc_synch_all(:,Sjs_to_do);

% average over each subject's two sessions:
sz = size(acc_hmm_all);
acc_hmm_all = squeeze(mean(reshape(acc_hmm_all,sz(1),sz(2),sz(3),2,sz(4)/2),4));
acc_synch_all = squeeze(mean(reshape(acc_synch_all,100,2,sz(4)/2),2));
nSj = sz(4)/2;

% set global parameters:
ifitmethod = 2; % this for the regression fit CV estimates outlined in HBM paper
acc_hmm = squeeze(acc_hmm_all(:,ifitmethod,:,:));

% cross validate timepoint by timepoint to determine K:
acc_hmm_CV_t = zeros(100,nSj);
for iSj=nSj
    SJ_in = setdiff(1:nSj,iSj);
    for it=1:100
        [~,K_best_t_CV(iSj,it)] = max(mean(mean(acc_hmm(it,:,SJ_in),1),3));
        acc_hmm_CV_t(it,iSj) = squeeze(acc_hmm(it,K_best_t_CV(iSj,it),iSj));
    end
end

figure('Position',[440 378 936 420]);
m = mean(acc_hmm_CV_t,2);
s = std(acc_hmm_CV_t,[],2);
t = 0.01:0.01:1;
shadedErrorBar(t,m,s./sqrt(nSj),{'Color','Blue','LineWidth',2},1); hold on;
m = mean(acc_synch_all,2);
s = std(acc_synch_all,[],2);
shadedErrorBar(t,m,s./sqrt(nSj),{'Color','Red','LineWidth',2},1); hold on;
plot4paper('Time (sec)','Accuracy');
line(t,zeros(length(t),1),'Color','Black','LineWidth',1,'LineStyle','--');


% and find significance:
thresh = 1;
[corrp,tstats] = osl_clustertf(permute(acc_hmm_CV_t-acc_synch_all,[2,3,1]),thresh);
corrp(corrp<0.95) = NaN;
corrp(corrp>0.95) = 1;
plot(t,0.1*corrp,'LineWidth',2,'Color',[0.5,0.5,0.5]);

h(1) = plot([NaN,NaN],[1 1],'Color','Blue','LineWidth',2);
h(2) = plot([NaN,NaN],[1 1],'Color','Red','LineWidth',2);
legend(h,{'STRM-Classifier','Timepoint-by-timepoint LDA'},'Location','EastOutside')

print([figdir,'Acc_HMM_CV_t'],'-dpng');
%%
% and compare with sliding window:
clear acc_SW acc_SW_CV W_best_t_CV
for iSj = Sjs_to_do
    fnameout = accfile_SW(iSj);
    load(fnameout,'acc_LGS_slidingwindow');
    acc_SW(:,:,iSj) = acc_LGS_slidingwindow;
end

% cross validate to optimise window size:
for iSj=Sjs_to_do
    SJ_in = setdiff(Sjs_to_do,iSj);
    for it=1:100
        [~,W_best_t_CV(iSj,it)] = max(mean(acc_SW(it,:,SJ_in),3),[],2);
        acc_SW_CV(it,iSj) = acc_SW(it,W_best_t_CV(iSj,it),iSj);
    end
end

figure('Position',[440 378 936 420]);
m = mean(acc_hmm_CV_t,2);
s = std(acc_hmm_CV_t,[],2);
t = 0.01:0.01:1;
shadedErrorBar(t,m,s./sqrt(nSj),{'Color','Blue','LineWidth',2},1); hold on;
m = mean(acc_synch_all,2);
s = std(acc_synch_all,[],2);
shadedErrorBar(t,m,s./sqrt(nSj),{'Color','Red','LineWidth',2},1); hold on;
plot4paper('Time (sec)','Accuracy');
line(t,zeros(length(t),1),'Color','Black','LineWidth',1,'LineStyle','--');
m = mean(acc_SW_CV,2);
s = std(acc_SW_CV,[],2);
t = 0.01:0.01:1;
shadedErrorBar(t,m,s./sqrt(nSj),{'Color','Green','LineWidth',2},1); hold on;
legend('STRM-Regression','Timepoint-by-timepoint regression','Sliding Window regression');
print([figdir,'Acc_HMM_vsSW_CV_t'],'-dpng');

%%
% and plot accuracy vs other ML classifiers

clear acc;
for iSj=Sjs_to_do
    filenamein = accsynchfile(iSj);
    load(filenamein);
    acc(:,:,iSj) = squeeze(acc_LGS_synchronous);
end
acc_synch = acc(:,1:3,:);
acc_synch = squeeze(mean(reshape(acc_synch,100,3,2,nSj),3));

acc_synch = cat(2,permute(acc_synch_all,[1,3,2]),acc_synch);
acc_synch = cat(2,permute(acc_hmm_CV_t,[1,3,2]),acc_synch);
acc_all_t = squeeze(mean(acc_synch,1));

figure();
errorbar([1:5],mean(acc_all_t'),std(acc_all_t')./sqrt(nSj),'*','LineWidth',2)

xlim([0.5,5.5]);
set(gca,'XTick',[1:5]);
labels = {'STRM-Regression','Time-Aligned LGS','Sliding Window LGS','LinearRegression','Linear SVM regression','RBF SVM Regression'};
set(gca,'XTickLabel',labels);
set(gca,'XTickLabelRotation',45);
plot4paper('Classifier','Accuracy');
yl = ylim();
clear paireddiff;
for i=1:4
    [~,pvals_HMM(i)] = ttest(acc_all_t(1,:)-acc_all_t(1+i,:));hold on;
    if pvals_HMM(i)<1e-5
        plot(i+1,yl(2)-0.05*(diff(yl)),'k*','LineWidth',2);
    end
    paireddiff(:,i) = acc_all_t(1,:)-acc_all_t(1+i,:);
end

line([1.5,1.5],yl,'Color', 'black');
print([figdir,'Acc_HMM_vs_MLClassifiers'],'-dpng');

%% Figure 2: Spatial activity maps:

figdir = [data_dir_foraging,'Figure2/'];
if ~isdir(figdir)
    mkdir(figdir);
end

load([data_dir_foraging,'UnloadingMatrices.mat']);
for iSj=Sjs_to_do
    load(modelfile(iSj));
    load(datafile(iSj));
    mod = model_LGS_HMM{iK};
    figure('Position',[10 63 1431 735]);
    K = size(mod.beta,3);
    % note in beta: first row is intercept!
    for ik=1:K
        subplot(ceil(K/2),K,ik);
        beta_intercept = mod.beta(1,:,ik)*UM{iSj};
        beta_intercept(1,60:62) = zeros(1,3);
        
        EEG_foraging_plot(beta_intercept',GLMinfo);
        subplot(ceil(K/2),K,ik+K);
        beta = mod.beta(2,:,ik)*UM;
        if EOGregressed
            beta(1,60:62) = zeros(1,3);
        end
        EEG_foraging_plot(beta',GLMinfo);
        beta_all(:,:,ik,iSj) = [beta_intercept;beta];
    end
    print([figdir,'Topoplot_Sj',int2str(iSj)],'-depsc');
    close all;
end

beta_all_m = mean(beta_all,4);
figure('Position',[10 63 1431 735]);
K = size(mod.beta,3);
% note in beta: first row is intercept!
for ik=1:K
    subplot(ceil(K/2),K,ik);
    beta_intercept = beta_all_m(1,:,ik);
    EEG_foraging_plot(beta_intercept',GLMinfo);
    subplot(ceil(K/2),K,ik+K);
    beta = beta_all_m(2,:,ik);
    EEG_foraging_plot(beta',GLMinfo);
end
print([figdir,'Topoplot_allsjmean'],'-depsc');

beta_all_m = mean(beta_all,4);
figure('Position',[10 63 1431 735]);
K = size(mod.beta,3);
% note in beta: first row is intercept!
clim1 = [min(squash(beta_all_m(1,1:59,:))),max(squash(beta_all_m(1,1:59,:)))];
clim2 = [min(squash(beta_all_m(2,1:59,:))),max(squash(beta_all_m(2,1:59,:)))];
for ik=1:K
    subplot(ceil(K/2),K,ik);
    beta_intercept = beta_all_m(1,:,ik);
    EEG_foraging_plot(beta_intercept',GLMinfo,clim1);
    subplot(ceil(K/2),K,ik+K);
    beta = beta_all_m(2,:,ik);
    EEG_foraging_plot(beta',GLMinfo,clim2);
end
print([figdir,'Topoplot_allsjmean_scaled'],'-depsc');


%% Figure 3: State timecourse modulation
figdir = [data_dir_foraging,'Figure3/']
mkdir(figdir);

Gam_all = [];
for iSj=Sjs_to_do
    
    load(modelfile(iSj));
    figure('Position', [440 96 253 702]);
    GM = gammaRasterPlot(Gamma_LGS{iK},100);
    print([figdir,'Rasterplot_Sj',int2str(iSj)],'-depsc');
    nregressors = 1;
    Gam_all = [Gam_all;Gamma_LGS{iK}];
    Gammean_sj(:,:,iSj) = mean(reshape(Gamma_LGS{iK},100,size(Gamma_LGS{iK},1)/100,K),2);
end
figure('Position', [440 96 253 702]);
GM = gammaRasterPlot(Gam_all,100);
set(gca,'XTick',[20:20:100]);
set(gca,'XTickLabel',[0.2:0.2:1]);
xlabel('Time (sec)');
print([figdir,'Rasterplot_allsjs'],'-depsc');

%%
P = 25; %prctile variation to show in shading
GamAll = reshape(Gam_all,[100,size(Gam_all,1)/100,K]);
M = squeeze(mean(GamAll,2));
M_upper = prctile(GamAll,100-P,2);
M_lower = prctile(GamAll,P,2);
figure('Position',[64 591 1377 207]);
t = 0.01:0.01:1;
colors = parula(K);
%plot(t,M,'LineWidth',2)
for ist=1:K
    patch([t';flipud(t')],[M_upper(:,:,ist);flipud(M_lower(:,:,ist))],colors(ist,:));
    hold on;
    alpha(0.3)  
end
for ist=1:K
    plot(t,M(:,ist),'Color',colors(ist,:),'LineWidth',2)
end
plot4paper('Time (sec)','Mean State Probability')
print([figdir,'MeanStateTimeCourse_prctile'],'-depsc');

% and plot as mean +/- ste:
Gammean_sj2 = squeeze(mean(reshape(Gammean_sj,100,K_CVSelect,2,23),3));
figure('Position',[64 591 1377 207]);
for ist=1:K_CVSelect
    %plot(t,M(:,ist),'Color',colors(ist,:),'LineWidth',2)
    shadedErrorBar(t',mean(Gammean_sj2(:,ist,:),3),std(Gammean_sj2(:,ist,:),[],3),{'Color',colors(ist,:),'LineWidth',2},0.5)
    hold on;
end
plot4paper('Time (sec)','Mean State Probability')
ylim([0,1]);
print([figdir,'MeanStateTimeCourse_std'],'-depsc');


%% plot behavioral correlations:

clear B_gmm pvals_sj p_ftest;
for iSj=Sjs_to_do
    fnamein = modelfile(iSj);
    load(fnamein)

    gamtemp = reshape(Gamma_LGS{iK},[100,size(Gamma_LGS{iK},1)/100,size(Gamma_LGS{iK},2)]);
    gamtemp = permute(gamtemp,[2,1,3]);
    
    % optionally filter out trials with highly deviant STCs:
    m = mean(gamtemp);
    resid = gamtemp - repmat(m,[size(gamtemp,1),1,1]);
    resid = sum(sum(resid.^2,3),2);
    outliers = resid > mean(resid)+2*std(resid);
    
    % compute transition times T:
    [~,vpath] = max(Gamma_LGS{iK},[],2);
    vpath = reshape(vpath,[100,size(Gamma_LGS{iK},1)/100]);
    clear Ttrans
    for i=1:K_options(iK)-1
        for itr = 1:size(vpath,2)
            Ttrans(itr,i) = find([vpath(:,itr);K+1]>i,1);
        end
    end
    Ttrans = demean(Ttrans,1) ./ 100;
    Ttrans(isnan(Ttrans))=0; % where state timecourses don't vary
    
    % setup subject specific design matrix:
    load(datafile(iSj));
    DM = DM(1:T(1):end,:);
    
    DM = normalise(DM);
    % decorrelate all regressors:
    DM_full = zeros(size(DM,1),size(DM,2));
    DM_full = decorrelateDesignMatrix(DM);
    DM_full = normalise(DM_full);

    % fit GLM:
    B_gmm(:,:,iSj) = pinv(DM_full)*Ttrans;
    
end

sz = size(B_gmm);
B_gmm = reshape(B_gmm,sz(1),sz(2),2,nSj);
B_gmm = reshape(mean(B_gmm,3),sz(1),sz(2),nSj);

DMlabels = {'Value','Leaving Time','AvgRRHist'};
for k=1:K-1
    xlabels{k} = [int2str(k), ' to ',int2str(k+1)];
end


figure('Position',[138 227 498 565]);
for k=1:size(B_gmm,1)
    subplot(ceil(0.5*(size(B_gmm,1))),2,k);
    B_mean = mean(B_gmm(k,:,:),3);
    bar(B_mean);hold on;
    m = max(abs(squash(mean(B_gmm,3))));
    for i=1:size(B_gmm,2)
        [~,pvals(k,i)] = ttest(B_gmm(k,i,:));
        if pvals(k,i) < 0.05/((size(B_gmm,1)-1)*(size(B_gmm,2)))
            plot(i-0.2,1.1*m,'k*','MarkerSize',15,'LineWidth',2);
            plot(i+0.2,1.1*m,'k*','MarkerSize',15,'LineWidth',2);
        elseif pvals(k,i)<0.05 % NO - only plot bonferroni corrected above
            %plot(i,1.1*m,'k*','MarkerSize',15,'LineWidth',1);
        end
    end
    
    ylim([-1.2,1.2]*m);
    plot4paper('State Transition','Timing modulation (sec)');
    title(DMlabels{k});
    set(gca,'XTick',[1:size(B_gmm,2)]);
    set(gca,'XTickLabel',xlabels);
    set(gca,'XTickLabelRotation',45);
    
end
print([figdir 'TransitionTimePredictiveModelling'],'-depsc')
