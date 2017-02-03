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

%% Perform cross-validation

options.cvmode = 3; 
[mcv,cv] = cvhmmmar (X,T,options);
disp(['CV-Log likelihood: ' num2str(mcv(1)) ])
disp(['CV-Fractional error: ' num2str(mcv(2)) ])

%% MAR spectra

options.Fs = 100; % Frequency in Hertzs
options.p = 0.01; % setting intervals of confidence
spectral_info = hmmspectramar(X,T,hmm,Gamma,options);

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







% %% Test sign flipping
% % this is useful in MEG source
% 
% for in = 1:N
%     ind = (1:T(in)) + sum(T(1:in-1));
%     X(ind,:) = X(ind,:) - repmat(mean(X(ind,:)),T(in),1);
%     X(ind,:) = X(ind,:) ./ repmat(std(X(ind,:)),T(in),1);
% end
% X_flip = X;
% Flips = binornd(1,0.2,N,ndim);
% for in = 1:N
%     if mean(Flips(in,:))>0.5 % we force to be more unflipped than flipped
%         Flips(in,:) = 1 - Flips(in,:);
%     end
%     for d = 1:ndim
%         ind = (1:T(in)) + sum(T(1:in-1));
%         if Flips(in,d), X_flip(ind,d) = -X_flip(ind,d);
%         end
%     end
% end
% 
% % r1 = zeros(ndim,ndim,3,N); r2 = zeros(ndim,ndim,3,N);
% % for in = 1:N
% %     ind = (1:T(in)) + sum(T(1:in-1));
% %     a1 = xcorr(X(ind,:),3,'coeff'); a2 = xcorr(X_Flip(ind,:),3,'coeff');
% %     for j=1:3
% %        r1(:,:,j,in) = reshape(a1(j,:),[ndim ndim]);
% %        r2(:,:,j,in) = reshape(a2(j,:),[ndim ndim]);
% %     end
% % end
% 
% options_sf = struct(); 
% options_sf.maxlag = 10; 
% options_sf.noruns = 10; 
% options_sf.nbatch = 3;
% options_sf.probinitflip = 0.1;
% options_sf.verbose = 1;
% 
% 
% %options_sf.Flips = Flips; 
% % options_sf.Flips(1,2) = ~options_sf.Flips(1,2); 
% % options_sf.Flips(4,1) = ~options_sf.Flips(4,1); 
% % options_sf.Flips(10,3) = ~options_sf.Flips(10,3);
% % options_sf.Flips(10,2) = ~options_sf.Flips(10,2);
% %options_sf.Flips = zeros(10,3); 
% %options_sf.Flips
% 
% [Flipshat,score] = findflip(X_flip,T,options_sf);
% X_unflip = flipdata(X_flip,T,Flipshat);
% 
% % Y = X_flip;
% % for in = 1:N
% % ind = (1:T(in)) + sum(T(1:in-1));
% % Y(ind,:) = Y(ind,:) - repmat(mean(Y(ind,:)),T(in),1);
% % Y(ind,:) = Y(ind,:) ./ repmat(std(Y(ind,:)),T(in),1);
% % end
% % 
% % C1 = getCovMats(Y,T,options_sf.maxlag,Flips);
% % C2 = getCovMats(Y,T,options_sf.maxlag,Flipshat);
% % [getscore(C1) getscore(C2)]
% % 
% % max(sum(abs(Flips(:)-Flipshat(:)))/length(Flipshat(:)) , ...
% %     1 - sum(abs(Flips(:)-Flipshat(:)))/length(Flipshat(:)) )
% 
% %% Test segment/subject alignment
% 
% X_task = randn(sum(T),size(X,2)); 
% for j=1:size(X,2),
%     X_task(:,j) = smooth(X_task(:,j),40);
% end
% 
% evoked_response = 4* mean(std(X_task)) * (-sin( (1:201) / 30))';
% weights = rand(1,size(X,2));
% window = [-100 100];
% events = [];
% flips = binornd(1,0.4,N,1);
% if mean(flips)>0.5, flips = 1-flips; end 
% 
% for i=1:N
%     e = round(T(i)*rand(3,1));
%     e = e( e>(-window(1)) & e<(T(i)-window(2)) );
%     events = [events; e+sum(T(1:i-1))];
%     for j=e'
%        X_task(sum(T(1:i-1)) + (j+window(1):j+window(2)), :) = ...
%            X_task(sum(T(1:i-1)) + (j+window(1):j+window(2)), :) + ...
%            repmat(evoked_response,1,size(X,2));
%     end
%     t = sum(T(1:i-1)) + (1:T(i));
%     if flips(i), X_task(t,:) = - X_task(t,:); end
% end   
% 
% [X_unflipped,flipshat] = aligntotask(X_task,T,events,window);
% 
% w=500;
% event1 = events(1); event2 = events(30);
% 
% 
% subplot(2,3,1)
% plot(X_task(-w+event1:w+event1,1)); ylim([-1.5 1.5]); 
% hold on; plot([400 400],[-1.5 1.5],'k'); hold off 
% subplot(2,3,2)
% plot(X_task(-w+event2:w+event2,1)) ; ylim([-1.5 1.5])
% hold on; plot([400 400],[-1.5 1.5],'k'); hold off 
% subplot(2,3,3)
% plot((X_task(-w+event1:w+event1,1) + X_task(-w+event2:w+event2,1) )/2) ; ylim([-1.5 1.5]) 
% hold on; plot([400 400],[-1.5 1.5],'k'); hold off 
% 
% subplot(2,3,4)
% plot(X_unflipped(-w+event1:w+event1,1)) ; ylim([-1.5 1.5])
% hold on; plot([400 400],[-1.5 1.5],'k'); hold off 
% subplot(2,3,5)
% plot(X_unflipped(-w+event2:w+event2,1)) ; ylim([-1.5 1.5])
% hold on; plot([400 400],[-1.5 1.5],'k'); hold off 
% subplot(2,3,6)
% plot((X_unflipped(-w+event1:w+event1,1) + X_unflipped(-w+event2:w+event2,1))/2 ) ; ylim([-1.5 1.5])
% hold on; plot([400 400],[-1.5 1.5],'k'); hold off 
% 
% 
% 
% 
% 
