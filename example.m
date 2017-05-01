%% Generate the data

clear
addpath(genpath('.'))
% Specify parameters
K = 4; % number of states
ndim = 3; % number of channels
N = 10; % number of trials
T = 10000*ones(N,1); % number of data points
epsilon = 0.1; %level of noise
StatePermanency = 100; % factor for the diagonal of the transtion matrix

hmmtrue = struct();
hmmtrue.K = K;
hmmtrue.state = struct();
% These 3 parameters essentially define a standard multivariate Gaussian distribution
hmmtrue.train.covtype = 'full';  
hmmtrue.train.zeromean = 0; 
hmmtrue.train.order = 0;

for k = 1:K
    hmmtrue.state(k).W.Mu_W = rand(1,ndim);
    hmmtrue.state(k).Omega.Gam_rate = randn(ndim) + eye(ndim);
    hmmtrue.state(k).Omega.Gam_rate = epsilon * 1000 * ...
        hmmtrue.state(k).Omega.Gam_rate' * hmmtrue.state(k).Omega.Gam_rate;
    hmmtrue.state(k).Omega.Gam_shape = 1000;
end

hmmtrue.P = ones(K,K) + StatePermanency * eye(K); %rand(K,K);
for j=1:K,
    hmmtrue.P(j,:)=hmmtrue.P(j,:)./sum(hmmtrue.P(j,:));
end;
hmmtrue.Pi = ones(1,K); %rand(1,K);
hmmtrue.Pi = hmmtrue.Pi./sum(hmmtrue.Pi);

[X,T,Gammatrue] = simhmmmar(T,hmmtrue,[]);
for d=1:ndim, X(:,d) = smooth(X(:,d)); end % introduce some time dependencies

%% Basic HMM-MAR run

options = struct();
options.K = 2; 
options.covtype = 'diag'; % model just variance of the noise
options.order = 6; % MAR order 6
options.zeromean = 0; % model the mean
options.tol = 1e-8;
options.cyc = 25;
options.inittype = 'hmmmar';
options.initcyc = 5;
options.initrep = 2;
options.verbose = 1;

[hmm, Gamma,~, ~, ~, ~, fehist] = hmmmar(X,T,options);
plot(fehist)

% re-estimate the viterbi path
[Path,Xi] = hmmdecode(X,T,hmm,1);

%% Perform cross-validation

options.cvmode = 3; 
[mcv,cv] = cvhmmmar (X,T,options);
disp(['CV-Log likelihood: ' num2str(mcv(1)) ])
disp(['CV-Fractional error: ' num2str(mcv(2)) ])

%% MAR spectra

options.Fs = 100; % Frequency in Hertzs
options.p = 0.01; % setting intervals of confidence
spectral_info = hmmspectramar(X,T,[],Gamma,options);

figure
cols = {'b','g','r'};
for k=1:2
    subplot(1,2,k)
    hold on
    for j=1:ndim
        plot(spectral_info.state(k).f,spectral_info.state(k).psd(:,j,j),cols{j})
    end
    hold off
    ylabel('Power (a.u.)','FontSize',16)
    xlabel('Frequency (Hz)','FontSize',16)
    title(['State ' num2str(k)],'FontSize',16)
    set(gca,'FontSize',16)
end

%% MT spectra

options_mt = struct('Fs',100);
options_mt.fpass = [1 48];
options_mt.tapers = [4 7];
options_mt.p = 0;
options_mt.win = 500;
%options_mt.to_do = [1 0]; % just coherence, not PDC
spectral_info = hmmspectramt(X,T,Gamma,options_mt);

figure
cols = {'b','g','r'};
for k=1:2
    subplot(1,2,k)
    hold on
    for j=1:ndim
        plot(spectral_info.state(k).f,spectral_info.state(k).psd(:,j,j),cols{j})
    end
    hold off
    ylabel('Power (a.u.)','FontSize',16)
    xlabel('Frequency (Hz)','FontSize',16)
    title(['State ' num2str(k)],'FontSize',16)
    set(gca,'FontSize',16)
end


