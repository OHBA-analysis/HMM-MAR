function ehmm = updateOmega_ehmm(ehmm,Gamma,residuals,XX,XXGXX,XW,Tfactor)
% not implemented for anything else than uniquediag

if nargin < 6, Tfactor = 1; end

ehmm.K = ehmm.K + 1;
ehmm = updateOmega(ehmm,Gamma,residuals,XX,XXGXX,XW,Tfactor);
ehmm.K = ehmm.K - 1;

% K = ehmm.K; ndim = ehmm.train.ndim;
% if nargin < 5, Tfactor = 1; end
% T = size(XX,1); 
%  
% setstateoptions;
% 
% [Xhat,XXstar] = computeStateResponses(XX,ehmm,Gamma,1:K+1);
% 
% e = (residuals(:,regressed) - Xhat(:,regressed)).^2;
% 
% swx2 = zeros(size(XX,1),ndim);
% for n = 1:ndim
%     if ~regressed(n), continue; end
%     Sind_all = repmat(Sind(:,n),K+1,1) == 1;
%     tmp = XXstar(:,Sind_all) * ehmm.state_shared(n).S_W(Sind_all,Sind_all);
%     swx2(:,n) = sum(tmp .* XXstar(:,Sind_all),2);
% end
% 
% ehmm.Omega.Gam_rate(regressed) = ehmm.prior.Omega.Gam_rate(regressed) + ...
%     0.5 * Tfactor * sum(e + swx2(:,regressed)); 
% ehmm.Omega.Gam_shape = ehmm.prior.Omega.Gam_shape + 0.5 * Tfactor * T;

end
