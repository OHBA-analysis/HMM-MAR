function [newstatetimecourse,stats] = fitEquivUnsupervisedModel(X_train,Gamma,X_test,T_train,T_test)
% Given a precomputed TUDA model, this function estimates an equivalent
% unsupervised model - that is, the distribution oer only the data that
% best describes the state timecourse
%
% INPUT
% X_train: Data that the TUDA\TUCA model was trained on
% Gamma:   Inferred TUDA\TUCA model state time course
% X_test:  Test set data for which state timecourses are to be estimated
% T_train: vector of trial lengths for X_train
% T_test:  vector of trial lengths for X_test
%
%
% OUTPUT
% newstatetimecourse: estimated state tiemcourse for the test data inputted
%
% Author: Cam Higgins, OHBA, University of Oxford (2018)
smoothoutput = true;

interceptchannels = find(std(X_train)==0);
X_train(:,interceptchannels) = [];
X_test(:,interceptchannels) = [];

options=struct();
%options.Fs=100;
options.K = size(Gamma,2); 
options.covtype = 'full'; % model full covariance of the noise per state
options.order = 0; % MAR order  0 - gaussian obs model
options.AR=0;
options.zeromean = 0; % model the mean
options.tol = 1e-8;
options.cyc = 5;
options.inittype = 'hmmmar';
options.verbose = 0;
options.useParallel = 0;
options.standardise = 0; %do not standardise
options.dropstates=false; % need to switch this off due to low number of data points
t_trial = T_train(1);
options.Gamma=Gamma;
options.updateGamma=0;
options.Pstructure = eye(options.K) + diag(ones(1,options.K-1),1);
options.Pistructure = zeros(1,options.K);
options.Pistructure(1)=1;
[hmm_unsup, Gamma_unsup,~, ~, ~, ~, fehist] = hmmmar(X_train,T_train,options);

options = rmfield(options,'Gamma');
options.hmm=hmm_unsup;
options.updateObs=0;
options.updateGamma=1;
options.cyc=5;

options.Pstructure = eye(options.K) + diag(ones(1,options.K-1),1);
options.Pistructure = zeros(1,options.K);
options.Pistructure(1)=1;

% setup semisupervised structure so all trials end in last state:
data = [];
data.X = X_test;
data.C = nan(size(X_test,1),options.K);
data.C(t_trial:t_trial:end,:) = repmat([zeros(1,options.K-1),1],length(T_test),1);

if nargin>2
    [unsup2, newstatetimecourse,~, ~, ~, ~, ~] = hmmmar(data,T_test,options);
else
    newstatetimecourse=[];
end


% final step, just smooth state timecourses to avoid jagged inference
% edges:
if smoothoutput
    kernelWidth = t_trial / (3 * options.K);
    x = -t_trial:t_trial; 
    gaussKernel = 1 / sqrt(2 * pi) / kernelWidth * exp(-x.^2/kernelWidth^2/2);
    gaussKernel = gaussKernel./max(gaussKernel);
    t_0 = 0;
    for itr = 1:length(T_test)
        Gamtr = zeros(T_test(itr),options.K);
        for ik = 1:options.K
            temp = newstatetimecourse(t_0 + [1:T_test(itr)],ik);
            Gamtr(:,ik) = conv(temp,gaussKernel,'same');
        end
        Gamtr = Gamtr./repmat(sum(Gamtr,2),1,options.K);
        newstatetimecourse(t_0 + [1:T_test(itr)],:) = Gamtr;
        t_0 = t_0 + T_test(itr);
    end
end

end