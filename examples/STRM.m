%% This script contains the analysis presented in Higgins et al 2021:
% "SpatioTemporally Resolved MVPA methods for M/EEG"
% 
% This script is organised in order to be most illustrative of the method;
% it begins by simulating data and fitting the model to demonstrate how the
% model works in simple 2D examples (these simulations are those presented
% in the SI of Higgins et al 2021); it then contains the analysis presented
% in the paper for the two datasets indicated. These datasets will be made
% publicly available upon acceptance and links to the data included below.
%
% Part 1: Simulations for STRM-Classification model
% Part 2: Simulations for STRM-Regression model
% Part 3: Analysis applied to categorical visual stimulus data
% Part 4: Analysis pipeline for continuous value EEG data


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
X = [ones(N*T,1),[ones(N*T/2,1);zeros(N*T/2,1)],[zeros(N*T/2,1);ones(N*T/2,1)]];

% generate data itself:
Y = zeros(N*T,P);
for itrial=1:N
    t_offset = (itrial-1)*T;
    for iT = 1:T
        Y(t_offset+iT,:) = mvnrnd(X(t_offset+iT,:)*W{find(z(t_offset+iT,:),1)},Sigma{find(z(t_offset+iT,:),1)});
    end
end
% scatter plot:
figure('Position', [6 478 1885 620]);
subplot(1,3,1);
scatter(Y(X(:,2)==1,1),Y(X(:,2)==1,2),10,cols{1});hold on;
scatter(Y(X(:,3)==1,1),Y(X(:,3)==1,2),10,cols{2});
plot4paper('Channel 1','Channel 2');
title('Scatter plot of all data generated')
legend('Class 1','Class 2')
% plot some data:
t = 1:T;
subplot(2,3,2);
itrial = N/2;t_offset = (itrial-1)*T;
plot(t,Y(t_offset + [1:T],:),'LineWidth',2);
title('Sample trial: Class 1');
plot4paper('Time','Channel signal');
z_thistrial = find(z(t_offset + [1:iT],2),1);
hold on;
plot([z_thistrial,z_thistrial],ylim,'k--')
legend({'Channel 1','Channel 2','Transition time'},'Location','EastOutside');
subplot(2,3,5);
itrial = N;t_offset = (itrial-1)*T;
plot(t,Y(t_offset + [1:T],:),'LineWidth',2);
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
[STRMmodel,STRMstates] = tucatrain(Y,X(:,2:3),repmat(T,N,1),options);

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
        scatter(Y(vpath(:,k)==1 & X(:,1+iclass)==1,1),Y(vpath(:,k)==1 & X(:,1+iclass)==1,2),10,cols{iclass})
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
        contour(XG,YG,ZG');hold on;
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
        contour(XG,YG,ZG');hold on;
    end
    a(1) = plot(W_exp_k(1,1),W_exp_k(1,2),'+','MarkerSize',20,'MarkerFaceColor',cols{1},'LineWidth',5,'Color',cols{1});
    hold on;
    a(2) = plot(W_exp_k(2,1),W_exp_k(2,2),'+','MarkerSize',20,'MarkerFaceColor',cols{2},'LineWidth',5,'Color',cols{2})
    legend(a,{'Class 1','Class 2'});
    title(['Ground truth dist for state ',int2str(k)])
    plot4paper('Channel 1','Channel 2')
end

print([figdir,'STRMClassification_Fig3B'],'-dpng')


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

clear mu Sigma X;
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
X = randn(nTr,1);
% simulate state timecourses:
Y = zeros(T,nTr,2);
Gamma_true = zeros(T,nTr,2);

for itr = 1:nTr
    t_stateswitch = floor(rand(1)*48) + 1;%round(0.5*rand(1)*T + T/2);
    Gamma_true(1:floor(t_stateswitch),itr,1) = 1;
    Gamma_true((t_stateswitch+1):end,itr,2) = 1;
    %simulate data in each state:
    Y(1:t_stateswitch,itr,:) = repmat(permute(mu{1}*X(itr,:)+offset{1},[1,3,2]),t_stateswitch,1,1);
    %Y(:,itr,:) = repmat(permute(mu{1}*X(itr,:)+offset{1},[1,3,2]),T,1,1);
    Y((t_stateswitch+1):end,itr,:) = repmat(permute(mu{2}*X(itr,:)+offset{2},[1,3,2]),T-t_stateswitch,1,1);
end
Y = reshape(Y,[T*nTr,ndim]);
Gamma_true = reshape(Gamma_true,[T*nTr,ndim]);
X = repelem(X,T,1);

% finally, add noise:
for t=1:T*nTr
    Y(t,:) = Y(t,:) + mvnrnd([0;0],Sigma{find(Gamma_true(t,:))});
end


% plot for sanity check - 
figure();
for k=1:size(X,1)
    col = min([max([0,(X(k)+1.5) ./3]),1]);
    col = col*[1,0,0.75];
    plot(Y(k,1),Y(k,2),'*','Color',col);hold on;
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
for k=1:size(X,1)
    col = (X(k) - min(X)) ./ (max(X)-min(X));
    col = col*[1,0,0.75];
    plot(Y(k,1),Y(k,2),'*','Color',col);hold on;
end
colmap = [0:0.01:1]'*[1,0,0.75];
colormap(colmap);
cb = colorbar;
TB = ([-1.5:0.5:1.5]- min(X)) ./ (max(X)-min(X));
set(cb,'Ticks',TB)
set(cb,'TickLabels',[-1.5:0.5:1.5]);
plot4paper('Channel 1','Channel 2');
title('Scatter plot of all data generated')
% plot some data:
t = 1:T;
subplot(2,3,2);
itrial = randi(N/2);t_offset = (itrial-1)*T;
plot(t,Y(t_offset + [1:T],:),'LineWidth',2);
ylim([-3,3]);
title(['Sample trial: X=',num2str(X(t_offset+1),3)]);
plot4paper('Time','Channel signal');
z_thistrial = find(Gamma_true(t_offset + [1:iT],2),1);
hold on;
plot([z_thistrial,z_thistrial],ylim,'k--');
legend({'Channel 1','Channel 2','Transition time'},'Location','EastOutside');
subplot(2,3,5);
itrial = N;t_offset = (itrial-1)*T;
plot(t,Y(t_offset + [1:T],:),'LineWidth',2);
ylim([-3,3]);
title(['Sample trial: X=',num2str(X(t_offset+1),3)]);
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
[STRMmodel,STRMstates] = tucatrain(Y,X,repmat(T,N,1),options);

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
    for t=1:size(X,1)
        col = min([max([0,(X(t)+1.5) ./3]),1]);
        col = col*[1,0,0.75];
        if vpath(t,k)==1
            plot(Y(t,1),Y(t,2),'*','Color',col);hold on;
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
        contour(XG,YG,ZG');hold on;
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
        contour(XG,YG,ZG');hold on;
    end
    h = plot([W_exp_k(:,1)],[W_exp_k(:,2)],'k-','LineWidth',1.5);hold on;
    plot([W_exp_k(2,1)],[W_exp_k(2,2)],'k>','LineWidth',1.5);
    hold on;
    title(['Ground truth dist for state ',int2str(k)])
    plot4paper('Channel 1','Channel 2')
end

print([figdir,'STRMReg_Fig3B'],'-dpng')

%% PART THREE: Visual MEG stimuli

fulldir = '/Users/chiggins/data/STRMData/MEGData/'; % project directory where data is stored
datafile = @(sjnum) [fulldir,'FLISj',int2str(sjnum),'.mat'];


%% Step 1: Full CV analysis estimating accuracies:

K_options = [4:2:22]; % Range of parameter values for K to test
Sjs_to_do = [1];
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


%% PART 4: EEG Data analysis

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
    load(fname,'data','DM','T','varexpl');
    
    Y = DM(:,1); % this is the stimulus value
    
    % and decode using LDA classifiers:
    rng(1); %ensure cross validation consistent
    [acc_LGS_synchronous] = standard_decoding(data,Y,T,options_LGS);

    for iK = 1:length(K_options)
        K = K_options(iK);
        options_LGS.K=K;
        rng(1); %ensure cross validation consistent
        [~,acc_LGS_HMM(:,:,:,iK,:),~,~,~,K_acc_LGS_HMM(1:K_options(iK),:,:,iK,:)] = tudacv(data,Y,T,options_LGS);
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
    acc_LGS_synchronous = zeros(100,1,3,2);
    
    fname = datafile(iSj);
    clear data DM T
    load(fname,'data','DM','T');
    
    Y = DM(:,1);
    
    rng(1); %ensure cross validation consistent
    
    options_reg.intercept = 1;
    acc_LGS_synchronous(:,1,1,:) = standard_decoding(data,Y,T,options_reg);

    options_regSVM = options_reg;
    options_regSVM.SVMreg = true;
    options_regSVM.SVMkernel = 'linear';
    acc_LGS_synchronous(:,1,2,:) =  standard_decoding(data,Y,T,options_regSVM);
    
    options_regSVM.SVMkernel = 'rbf';
    acc_LGS_synchronous(:,1,3,:) = standard_decoding(data,Y,T,options_regSVM);
    
    
    fnameout = accsynchfile(iSj);
    save(fnameout,'acc_LGS_synchronous','synch_class_labels');
end

%% compare to equivalent sliding window methods:

options_LGS = [];
options_LGS.NCV = 10;
options_LGS.intercept = 1;   
options_LGS.encodemodel = true;
options_LGS.covtype = 'full';
options_LGS.accuracyType = 'all';


windows = 1:10;
accfile_SW = @(sjnum) [data_dir_foraging,'Acc_SW_Sj',int2str(sjnum),'.mat'];

for iSj = Sjs_to_do
    acc_LGS_slidingwindow = zeros(100,length(windows),2);
    
    fprintf(['Decoding Subject ',int2str(iSj),'\n']);
    
    fname = datafile(iSj);
    clear data DM T varexpl
    load(fname,'data','DM','T','varexpl');
    
    Y = DM(:,1);
    
    rng(1); %ensure cross validation consistent
    
    % apply sliding window averaging
    for iwindow=1:length(windows)
        options_LGS.slidingwindow = windows(iwindow);
        acc_LGS_slidingwindow(:,iwindow,:) = squeeze(standard_decoding(data,Y,T,options_LGS));
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
    Y = DM(:,1); % note Y is just stimulus value
    
    model_LGS_HMM = cell(length(K_options),1);
    for iK = 1:length(K_options)
        K = K_options(iK);
        options_LGS.K=K;
        options_LGS.verbose=false;
        rng(1); %ensure cross validation consistent
        [model_temp,Gamma_LGS{iK}] = tudatrain(data,Y,T,options_LGS);
        model_LGS_HMM{iK}.beta = tudabeta(model_temp);
        model_LGS_HMM{iK}.Sigma= tudaomega(model_temp);
    end
    fnameout = modelfile(iSj);
    save(fnameout,'model_LGS_HMM','Gamma_LGS','K_options','options_LGS');
end