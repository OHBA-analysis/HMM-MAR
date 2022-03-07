function ehmm = obsupdate_ehmm(Gamma,ehmm,residuals,XX,Tfactor)
%
% Update observation model
%
% INPUT
% X             observations
% T             length of series
% Gamma         p(state given X)
% ehmm           ehmm data structure
% residuals     in case we train on residuals, the value of those.
%
% OUTPUT
% ehmm           estimated ehmm model
%
% Author: Diego Vidaurre, OHBA, University of Oxford / Aarhus Univ (2022)

% Some stuff that will be later used
if nargin<6, Tfactor = 1; end

K = ehmm.K;

Gamma = [Gamma prod(1-Gamma,2)];
Gamma = rdiv(Gamma,sum(Gamma,2));  

XXGXX = cell(K,1);
for k = 1:K+1
    XXGXX{k} = bsxfun(@times, XX, Gamma(:,k))' * XX;
end

% autoregression coefficients
[ehmm,XW] = updateW_ehmm(ehmm,Gamma,residuals,XX,XXGXX,Tfactor);

% Omega
ehmm = updateOmega_ehmm(ehmm,Gamma,residuals,XX,XXGXX,XW,Tfactor);
% 
% % autoregression coefficient priors
ehmm = updateSigma_ehmm(ehmm); % sigma - channel x channel coefficients
ehmm = updateAlpha_ehmm(ehmm); % alpha - one per order

end









% function ehmm = updateW_here(ehmm,Gamma,residuals,XX)
% 
% K = size(Gamma,2); np = size(XX,2);  
% ndim = size(ehmm.state(end).W.Mu_W,2); 
% noGamma = prod(1-Gamma,2);
% residuals0 = residuals; 
% m0 = zeros(size(residuals0));
% 
% for n = 1:ndim
%     m0(:,n) = bsxfun(@times,XX * ehmm.state(end).W.Mu_W(:,n),noGamma);
%     residuals(:,n) = residuals(:,n) - m0(:,n);
% end
%     
% %Gamma = [Gamma noGamma];
% X = zeros(size(XX,1),np * K);
% for k = 1:K
%    X(:,(1:np) + (k-1)*np) = bsxfun(@times, XX, Gamma(:,k)); 
% end
% % estimation
% for n = 1:ndim
%     ehmm.state_shared(n).Mu_W = ...
%         pinv(X) * residuals(:,n);
% end
% 
% % meand = zeros(size(XX,1),ndim);
% % for n = 1:ndim
% %      W = ehmm.state_shared(n).Mu_W;
% %     meand(:,n) = meand(:,n) + X * W;
% % end
% % d = residuals - meand;
% % mean(d.^2) 
% 
% 
% end


% function [e,meand] = getError_here(Gamma,hmm,residuals,XX)
% 
% C = hmm.Omega.Gam_shape ./ hmm.Omega.Gam_rate;
% meand = computeStateResponses_here(XX,hmm,Gamma);
% d = residuals - meand;
% 
% [ mean(d.^2) ]
% 
% Cd = bsxfun(@times, C, d)';
% dist = zeros(size(residuals,1),1);
% for n = 1:size(residuals,2)
%     dist = dist + 0.5 * (d(:,n).*Cd(n,:)');
% end
% e = sum(dist); 
% 
% end
% 
% 
% function meand = computeStateResponses_here(XX,ehmm,Gamma)
% 
% K = size(Gamma,2); np = size(XX,2);  
% ndim = size(ehmm.state(end).W.Mu_W,2); 
% noGamma = prod(1-Gamma,2);
% meand = zeros(size(XX,1),ndim);
% for n = 1:ndim
%     meand(:,n) = meand(:,n) + ...
%         bsxfun(@times,XX * ehmm.state(end).W.Mu_W(:,n),noGamma);
% 
% end
% % meand = zeros(size(XX,1),ndim);
% %Gamma = [Gamma noGamma];
% X = zeros(size(XX,1),np * K);
% for k = 1:K
%    X(:,(1:np) + (k-1)*np) = bsxfun(@times, XX, Gamma(:,k)); 
% end
% for n = 1:ndim
%     %W = [ehmm.state_shared(n).Mu_W; ehmm.state(end).W.Mu_W(:,n)];
%     W = ehmm.state_shared(n).Mu_W;
%     %meand(:,n) = meand(:,n) + X * ehmm.state_shared(n).Mu_W;
%     meand(:,n) = meand(:,n) + X * W;
% end
% 
% end
