
%% Generate some data
addpath(genpath('.'))

K =3; p = 10; q = 3; 
N = 200;
ttrial = 150;

% random regression coefficients
beta = randn(p,q,K);

X = randn(ttrial,N,p); % data
Y = zeros(ttrial,N,q); % stimulus
g = zeros(ttrial,N,K); % state time courses
T = ttrial * ones(N,1); 
for n = 1:N
    t = 1; k = 1; 
    while t<ttrial
        l = round(20 * rand) + 10;
        t2 = t+l-1; 
        if t2>ttrial, t2 = ttrial; end
        g(t:t2,n,k) = 1;
        Y(t:t2,n,:) = squeeze(X(t:t2,n,:)) * beta(:,:,k) + 0.1 * randn(t2-t+1,q); 
        t = t + l;
        k = k + 1;
        if k > K, k = 1; end
    end
end
    
X = reshape(X,[ttrial*N p]);
Y = reshape(Y,[ttrial*N q]);

options = struct();
options.K = K;
options.verbose = 1;
[tuda,Gamma] = tudatrain(X,Y,T,options);

% if the distinction between the 3 models is not very clear, it's because
% two of the columns of beta turned out to be too similar. You might just
% want to rerun

figure; 
subplot(2,2,1); plot(squeeze(mean(reshape(Gamma,[ttrial N K]),2)))
subplot(2,2,3); plot(squeeze(mean(g,2)))
subplot(2,2,2); plot(Gamma(1:ttrial,:))
subplot(2,2,4); plot(squeeze(g(:,1,:)))


%% 

options.NCV = 5;
options.lossfunc = 'quadratic';
R2 = tudacv(X,Y,T,options);
figure(4);clf(4)
plot(R2)

%% Just another example

% Just another way to generate data
% X = randn(10000,5); T = 1000 * ones(10,1); 
% for j = 1:5, X(:,j) = smooth(X(:,j)); end
% beta = randn(5,3);
% change_models = zeros(10,2);
% for j = 1:2
%     for i = 1:10
%         change_models(i,j) = j*333 + (round(100*rand) - 50); 
%     end
% end
% change_models = [ones(10,1) change_models 1000*ones(10,1)];
% Y = zeros(10000,1);
% for j = 1:3
%     for i = 1:10
%         t = change_models(i,j):change_models(i,j+1)-1;
%         t = t + (i-1)*1000;
%         Y(t) = X(t,:) * beta(:,j) + 0.25 * randn(length(t),1);
%     end
% end

% Gamma3 = reshape(Gamma,[ttrial N K]);
% figure(1);clf(1)
% cols = ['b','r','g'];
% for i=1:10
%     subplot(5,2,i)
%     gamma = squeeze(Gamma3(:,i,:));
%     hold on
%     for k = 1:3
%         plot(gamma(:,k),cols(k),'LineWidth',4); xlim([1 1000]); ylim([-0.1 1.1])
%     end
%     for j=2:3, plot([change_models(i,j) change_models(i,j)],[-0.1 1.1],'k'); end
% end; clear Gamma3



