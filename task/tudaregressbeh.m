function [beh_hat,ev] = tudaregressbeh(Gamma,T,beh,npca,NCV,lambda)
% 
% Predicts behaviour using the state time courses obtained from TUDA 
% (or from the HMM)
%
% INPUTS
%
% Gamma: State time courses estimated from a TUDA or HMM model
% T: length of the trials
% beh: (trials by traits) matrix or vector of behavioural variables to
%       predict (e.g. reaction time)
% npca: how many PCA components to use to describe the state time courses
%       (default 10 by the number of states)
%       If -1, it uses instead the probability of transitioning per trial,
%       and does not perform PCA
% NCV: number of cross-validation folds (default 10)
% lambda: regularisation parameter for the estimation (default 1e-4)
% 
% OUTPUT
%
% beh_hat: cross-validation predicted behavioural variables
% ev: explained variance
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2019)

N = length(T); K = size(Gamma,2); q = size(beh,2);
if sum(T)>size(Gamma,1), T = T - 1; end
ttrial = T(1); 
if ~all(T==ttrial), error('All elements of T must be equal here'); end 
if length(beh)~=N, error('beh should have as many elements as trials'); end
if nargin<4, npca = 10*K; end
if nargin<5, NCV = 10; end
if nargin<6, lambda = 1e-4; end

if npca == -1
    G = zeros(N,K,K);
    for n = 1:N
       t = sum(T(1:n-1)) + (1:T(n));
       g = Gamma(t,:);
       G(n,:,:) = (g(1:end-1,:)' * g(2:end,:)) / T(n); 
    end
    G = reshape(G,[N K^2]);
else
    Gamma = reshape(permute(reshape(Gamma,[ttrial N K]),[2 1 3]),[N ttrial*K]);
    [~,G] = pca(Gamma,'NumComponents',npca);
end
    
RidgePen = lambda * eye(size(G,2));

c2 = cvpartition(N,'KFold',NCV);
c = struct();
c.test = cell(NCV,1);
c.training = cell(NCV,1);
for icv = 1:NCV
    c.training{icv} = find(c2.training(icv));
    c.test{icv} = find(c2.test(icv));
end; clear c2

beh_hat = zeros(N,q);

for icv = 1:NCV
    Ntr = length(c.training{icv}); Nte = length(c.test{icv});
    Gtrain = G(c.training{icv},:);
    Gtest = G(c.test{icv},:);
    mu = mean(beh(c.training{icv},:));
    behtrain = beh(c.training{icv},:) - repmat(mu,Ntr,1);
    beta = (Gtrain' * Gtrain + RidgePen) \ Gtrain' * behtrain;
    beh_hat(c.test{icv},:) = Gtest * beta + repmat(mu,Nte,1);
end

beh_0 = repmat(mean(beh),N,1);
ev = 1 - sum((beh_hat - beh).^2) ./ sum((beh - beh_0).^2);

end
