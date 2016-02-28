%% Synthetic data examples

clear
addpath(genpath('.'))
% Specify parameters
K = 4; % number of states
ndim = 3; % number of channels
N = 10; % number of trials
T = 1000*ones(N,1); % number of data points
epsilon = 0.1; %level of noise
StatePermanency = 100; % factor for the diagonal of the transtion matrix
%% Generate the data
hmmtrue = struct();
hmmtrue.K = K;
hmmtrue.state = struct();
hmmtrue.train.covtype = 'full';
hmmtrue.train.zeromean = 0;
hmmtrue.train.order = 0;
hmmtrue.train.orderoffset = 0;
hmmtrue.train.timelag = 1;
hmmtrue.train.exptimelag = 0;
hmmtrue.train.S =ones(ndim);
hmmtrue.train.Sind = ones(1,ndim);
hmmtrue.train.multipleConf = 0;

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
hmmtrue.Pi=hmmtrue.Pi./sum(hmmtrue.Pi);

[X,T,Gammatrue] = simhmmmar(T,hmmtrue,[]);
for d=1:ndim, X(:,d) = smooth(X(:,d)); end

%% Initial quick run to test that everything is ok

options = struct();
options.K = 2; 
options.covtype = 'full';
options.order = 6;
options.timelag = 2; 
options.DirichletDiag = 2; 
options.tol = 1e-12;
options.cyc = 100;
options.zeromean = 0;
options.inittype = 'GMM';
options.initcyc = 100;
options.initrep = 5;
options.verbose = 1;

[hmm, Gamma,~, ~, ~, ~, fehist] = hmmmar(X,T,options);
plot(fehist)
%% MAR spectra

options.Fs = 100; 
options.completelags = 1;
options.MLestimation = 1; 
options.order = 10; % increase the order

spectral_info = hmmspectramar(X,T,hmm,Gamma,options);
for k=1:2
    subplot(1,2,k)
    plot(spectral_info.state(k).f,spectral_info.state(k).psd(:,1,1),'k')
end

%% MT spectra

options_mt = struct('Fs',200);
options_mt.fpass = [1 48];
options_mt.tapers = [4 7];
options_mt.p = 0;
options_mt.win = 500;
options_mt.Gamma = Gamma;
%options_mt.to_do = [1 0]; % just coherence, not PDC
spectral_info = hmmspectramt(X,T,options_mt);
for k=1:2
    subplot(1,2,k)
    plot(spectral_info.state(k).f,spectral_info.state(k).psd(:,1,1),'k')
end
%% Train models on a grid of parameters

KK = 2:8;
DD = [2 10 20 100 200];
repetitions = 10;
options.verbose = 0;

FE = zeros(length(KK),length(DD),repetitions);
Ktrained = zeros(length(KK),length(DD),repetitions);
fefinal = Inf;

for r = 1:repetitions
    for idd=1:length(DD)
        dd = DD(idd);
        for ik=1:length(KK)
            k = KK(ik);
            fprintf('Rep %d; Using K = %d, DD=%d \n',r,k,dd);
            options.K = k;
            options.DirichletDiag = dd;
            [hmm, ~,~, ~, ~, ~, fehist] = hmmmar(X,T,options);
            FE(ik,idd,r) = fehist(end);
            Ktrained(ik,idd,r) = length(hmm.state);
            %if FE(ik,idd)<fefinal
            %    fefinal = FE(ik,idd);
            %    hmmfinal = hmm;
            %    Gammafinal = Gamma;
            %    Kfinal = k;
            %    DDfinal = dd;
            %end
        end
    end
    %fprintf('Selected K = %d; DD=%d \n',Kfinal,DDfinal);
end

% And final run with the selected parameters
meanFE = mean(FE,3);
meanKtrained = mean(Ktrained,3);
[~,I] = min(meanFE(:));
[I1,I2] = ind2sub(size(meanFE),I);
options.K = KK(I1);
options.DirichletDiag = DD(I2);
[hmm, Gamma,~, ~, ~, ~, fehist] = hmmmar(X,T,options);

save('example.mat')

%% Get frequency information of the last run

options_mt = struct('Fs',200);
options_mt.fpass = [1 48];
options_mt.tapers = [4 7];
options_mt.p = 0.05;
options_mt.win = 500;
options_mt.Gamma = Gamma;
%options_mt.to_do = [1 0]; % just coherence, not PDC
spectral_info = hmmspectramt(X,T,options_mt);

%% Plot results

addpath('../HMM-MAR-scripts/distributionPlot/')
load('example.mat')

figure(1)

subplot(1,2,1); 
imagesc(meanFE); colorbar; 
set(gca,'XTick',1:length(DD))
set(gca,'XTickLabel',DD,'FontSize',16)
set(gca,'YTick',1:length(KK))
set(gca,'YTickLabel',KK,'FontSize',16)
xlabel('Prior on diag(P)','FontSize',18)
ylabel('Initial K','FontSize',18)
title('Free Energy','FontSize',20)

subplot(1,2,2); 
imagesc(meanKtrained); colorbar; 
set(gca,'XTick',1:length(DD))
set(gca,'XTickLabel',DD,'FontSize',16)
set(gca,'YTick',1:length(KK))
set(gca,'YTickLabel',KK,'FontSize',16)
xlabel('Prior on diag(P)','FontSize',18)
ylabel('Initial K','FontSize',18)
title('Final K','FontSize',20)

Colors = {'b','g','r','m','k'};

figure(2); clf(2)
to_show = [1 2 3 4 5];
for d=to_show
    subplot(1,length(to_show),find(d==to_show));
    %boxplot(squeeze(FE(:,d,:))')
    hold on; 
    %distributionPlot(squeeze(FE(:,d,:))','xValues',KK);
    %for r=1:repetitions
    %    plot(KK,FE(:,d,r)','b','LineWidth',1)
    %end
    plot(KK,meanFE(:,d),'b','LineWidth',3)
    errorbar(KK,meanFE(:,d),std(squeeze(FE(:,d,:)),1,2) )
    xlim([KK(1) KK(end)])
    hold off
    title(strcat('dd=',num2str(DD(d))),'FontSize',18)
    xlabel('No. of states','FontSize',18)
    ylabel('FE','FontSize',18)
end
%xlabel('FE','FontSize',18)
%ylabel('K','FontSize',18)
%title('Free energy for the chosen dd','FontSize',20)

figure(3); clf(3)
subplot(2,1,1); 
plot(1:200,Gammatrue(101:300,:)); ylim([-.25 1.25])
xlabel('Time','FontSize',18)
ylabel('True State time courses','FontSize',18)
subplot(2,1,2); 
plot(1:200,Gamma(101:300,:)); ylim([-.25 1.25])
xlabel('Time','FontSize',18)
ylabel('Estimated state time courses','FontSize',18)


%% Test BigHMM algorithm 

% Big-HMM options
options_bighmm = struct();
% Specific BigHMM-MAR options
options_bighmm.K = 8;
options_bighmm.BIGNbatch = 2;
options_bighmm.BIGuniqueTrans = 1;
options_bighmm.BIGtol = 1e-7;
options_bighmm.BIGcyc = 100;
options_bighmm.BIGinitcyc = 1;
options_bighmm.BIGundertol_tostop = 5;
options_bighmm.BIGdelay = 5;
options_bighmm.BIGforgetrate = 0.75;
options_bighmm.BIGbase_weights = 0.9;
options_bighmm.BIGverbose = 1;
% HMM-MAR options
options_bighmm.covtype = 'full';
options_bighmm.order = 0;
options_bighmm.zeromean = 0;
options_bighmm.decodeGamma = 0;
options_bighmm.tol = 1e-7;
options_bighmm.cyc = 50;
options_bighmm.inittype = 'GMM';
options_bighmm.DirichletDiag = 2;
options_bighmm.initcyc = 5;
options_bighmm.initrep = 1;

TBig = {}; for n=1:length(T), TBig{n} = T(n); end
XBig = {}; for n=1:length(T), XBig{n} = X((1:T(n))+sum(T(1:n-1)),:); end

[hmm,markovTrans,fehist] = trainBigHMM(XBig,TBig,options_bighmm);
Gamma = decodeBigHMM(XBig,TBig,hmm,markovTrans,0);


%% Test sign flipping

for in = 1:N
    ind = (1:T(in)) + sum(T(1:in-1));
    X(ind,:) = X(ind,:) - repmat(mean(X(ind,:)),T(in),1);
    X(ind,:) = X(ind,:) ./ repmat(std(X(ind,:)),T(in),1);
end
X_Flip = X;
Flips = binornd(1,0.2,N,3);
for in = 1:N
    if mean(Flips(in,:))>0.5
        Flips(in,:) = 1 - Flips(in,:);
    end
    for d = 1:3
        ind = (1:T(in)) + sum(T(1:in-1));
        if Flips(in,d), X_Flip(ind,d) = -X_Flip(ind,d);
        end
    end
end

r1 = zeros(ndim,ndim,3,N); r2 = zeros(ndim,ndim,3,N);
for in = 1:N
    ind = (1:T(in)) + sum(T(1:in-1));
    a1 = xcorr(X(ind,:),3,'coeff'); a2 = xcorr(X_Flip(ind,:),3,'coeff');
    for j=1:3
       r1(:,:,j,in) = reshape(a1(j,:),[ndim ndim]);
       r2(:,:,j,in) = reshape(a2(j,:),[ndim ndim]);
    end
end

options_sf = struct(); 
options_sf.maxlag = 10; 
options_sf.noruns = 10; 
options_sf.nbatch = 3;
options_sf.probinitflip = 0.1;
options_sf.verbose = 1;


%options_sf.Flips = Flips; 
% options_sf.Flips(1,2) = ~options_sf.Flips(1,2); 
% options_sf.Flips(4,1) = ~options_sf.Flips(4,1); 
% options_sf.Flips(10,3) = ~options_sf.Flips(10,3);
% options_sf.Flips(10,2) = ~options_sf.Flips(10,2);
%options_sf.Flips = zeros(10,3); 
%options_sf.Flips

[Flipshat,score] = unflipchannels(X_Flip,T,options_sf);

%Flips
%Flipshat

Y = X_Flip;
for in = 1:N
ind = (1:T(in)) + sum(T(1:in-1));
Y(ind,:) = Y(ind,:) - repmat(mean(Y(ind,:)),T(in),1);
Y(ind,:) = Y(ind,:) ./ repmat(std(Y(ind,:)),T(in),1);
end

C1 = getCovMats(Y,T,options_sf.maxlag,Flips);
C2 = getCovMats(Y,T,options_sf.maxlag,Flipshat);
[getscore(C1) getscore(C2)]

max(sum(abs(Flips(:)-Flipshat(:)))/length(Flipshat(:)) , ...
    1 - sum(abs(Flips(:)-Flipshat(:)))/length(Flipshat(:)) )
