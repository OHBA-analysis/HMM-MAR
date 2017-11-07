addpath(genpath('.'))

X = randn(10000,5); T = 1000 * ones(10,1); 
for j = 1:5, X(:,j) = smooth(X(:,j)); end
beta = randn(5,3);
change_models = zeros(10,2);
for j = 1:2
    for i = 1:10
        change_models(i,j) = j*333 + (round(100*rand) - 50); 
    end
end
change_models = [ones(10,1) change_models 1000*ones(10,1)];
Y = zeros(10000,1);
for j = 1:3
    for i = 1:10
        t = change_models(i,j):change_models(i,j+1)-1;
        t = t + (i-1)*1000;
        Y(t) = X(t,:) * beta(:,j) + 0.25 * randn(length(t),1);
    end
end
    
options = struct();
options.K = 3;
options.verbose = 1;
[tuda,Gamma,GammaInit] = tudatrain(X,Y,T,options);

Gamma3 = reshape(Gamma,[1000 10 3]);
figure(1);clf(1)
cols = ['b','r','g'];
for i=1:10
    subplot(5,2,i)
    gamma = squeeze(Gamma3(:,i,:));
    hold on
    for k = 1:3
        plot(gamma(:,k),cols(k),'LineWidth',4); xlim([1 1000]); ylim([-0.1 1.1])
    end
    for j=2:3, plot([change_models(i,j) change_models(i,j)],[-0.1 1.1],'k'); end
end; clear Gamma3

% if the distinction between the 3 models is not very clear, it's because
% two of the columns of beta turned out to be too similar. You might just
% want to rerun

%% 

options.NCV = 5;
options.lossfunc = 'quadratic';
R2 = tudacv(X,Y,T,options);
figure(2);clf(2)
plot(R2)
