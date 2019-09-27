% For the real data experiments in:
%   Vidaurre et al (2018). Temporally unconstrained decoding reveals consistent 
%       but time-varying stages of stimulus processing. Cerebral Cortex
% check:
%   https://github.com/vidaurre/CC2018
%% Example for a continuous response that varies across time

addpath(genpath('.')) % HMM-MAR path

K = 3; p = 10; q = 3; 
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

options.NCV = 5;
[R2,acc_R2] = tudacv(X,Y,T,options);
figure
plot(mean(acc_R2,2)) 
% we lose accuracy as the trial progress because in the testing folds we don't know
% which state is the right one to use (which state should be used is based
% on both data and stimulus, but in the testing folds we don't have the
% stimulus, by definition)



