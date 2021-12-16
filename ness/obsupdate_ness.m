function ness = obsupdate_ness(T,Gamma,ness,residuals,XX,Tfactor)
%
% Update observation model
%
% INPUT
% X             observations
% T             length of series
% Gamma         p(state given X)
% hmm           hmm data structure
% residuals     in case we train on residuals, the value of those.
%
% OUTPUT
% hmm           estimated HMMMAR model
%
% Author: Diego Vidaurre, OHBA, University of Oxford

% Some stuff that will be later used
if nargin<6, Tfactor = 1; end

if isfield(ness.train,'distribution') && strcmp(ness.train.distribution,'logistic')
    error('Logistic regression not yet implemented for NESS')
end


% LK = [];
% for k = 1:3, LK = [LK obslike_ness(ness,Gamma,residuals,XX,k)]; end


% e1 = getError_here(Gamma,ness,residuals,XX);
% ness = updateW_here(ness,Gamma,residuals,XX);
ness = updateW_ness(ness,Gamma,residuals,XX,Tfactor); %_nobaseline
% e2 = getError_here(Gamma,ness,residuals,XX);
% e3 = getError_here(Gamma,ness2,residuals,XX);
%disp(num2str(e1-e2));



%ness.state_shared(1).Mu_W'

% LK0 = LK; LK = [];
% for k = 1:3, LK = [LK obslike_ness(ness,Gamma,residuals,XX,k)]; end
% sum(LK-LK0)
% mean(Gamma)

% Omega

% ness = updateOmega_ness(ness,Gamma,residuals,T,XX,Tfactor);
% 
% % autoregression coefficient priors
ness = updateSigma_ness(ness); % sigma - channel x channel coefficients
ness = updateAlpha_ness(ness); % alpha - one per order

end

function ness = updateW_here(ness,Gamma,residuals,XX)

K = size(Gamma,2); np = size(XX,2);  
ndim = size(ness.state(end).W.Mu_W,2); 
noGamma = prod(1-Gamma,2);
residuals0 = residuals; 
m0 = zeros(size(residuals0));

for n = 1:ndim
    m0(:,n) = bsxfun(@times,XX * ness.state(end).W.Mu_W(:,n),noGamma);
    residuals(:,n) = residuals(:,n) - m0(:,n);
end
    
%Gamma = [Gamma noGamma];
X = zeros(size(XX,1),np * K);
for k = 1:K
   X(:,(1:np) + (k-1)*np) = bsxfun(@times, XX, Gamma(:,k)); 
end
% estimation
for n = 1:ndim
    ness.state_shared(n).Mu_W = ...
        pinv(X) * residuals(:,n);
end

% meand = zeros(size(XX,1),ndim);
% for n = 1:ndim
%      W = ness.state_shared(n).Mu_W;
%     meand(:,n) = meand(:,n) + X * W;
% end
% d = residuals - meand;
% mean(d.^2) 


end


function [e,meand] = getError_here(Gamma,hmm,residuals,XX)

C = hmm.Omega.Gam_shape ./ hmm.Omega.Gam_rate;
meand = computeStateResponses_here(XX,hmm,Gamma);
d = residuals - meand;

[ mean(d.^2) ]

Cd = bsxfun(@times, C, d)';
dist = zeros(size(residuals,1),1);
for n = 1:size(residuals,2)
    dist = dist + 0.5 * (d(:,n).*Cd(n,:)');
end
e = sum(dist); 

end


function meand = computeStateResponses_here(XX,ness,Gamma)

K = size(Gamma,2); np = size(XX,2);  
ndim = size(ness.state(end).W.Mu_W,2); 
noGamma = prod(1-Gamma,2);
meand = zeros(size(XX,1),ndim);
for n = 1:ndim
    meand(:,n) = meand(:,n) + ...
        bsxfun(@times,XX * ness.state(end).W.Mu_W(:,n),noGamma);

end
% meand = zeros(size(XX,1),ndim);
%Gamma = [Gamma noGamma];
X = zeros(size(XX,1),np * K);
for k = 1:K
   X(:,(1:np) + (k-1)*np) = bsxfun(@times, XX, Gamma(:,k)); 
end
for n = 1:ndim
    %W = [ness.state_shared(n).Mu_W; ness.state(end).W.Mu_W(:,n)];
    W = ness.state_shared(n).Mu_W;
    %meand(:,n) = meand(:,n) + X * ness.state_shared(n).Mu_W;
    meand(:,n) = meand(:,n) + X * W;
end

end
