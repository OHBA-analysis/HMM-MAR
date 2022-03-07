function [ehmm,XW] = updateW_ehmm(ehmm,Gamma,residuals,XX,XXGXX,Tfactor,lambda)

if nargin < 6, Tfactor = 1; end
if nargin < 7, lambda = []; end

XW = zeros(size(residuals,1),size(residuals,2),ehmm.K+1);
[ehmm,XW(:,:,1:ehmm.K)] = updateW(ehmm,Gamma,residuals,XX,XXGXX,...
    Tfactor,1:ehmm.K,lambda);
XW(:,:,ehmm.K+1) = XX * ehmm.state(end).W.Mu_W;

K = size(Gamma,2); ndim = size(ehmm.train.S,1); np = size(XX,2);
for k = 1:K
    ind = (k-1)*np + (1:np);
    for n = 1:ndim
        ehmm.state_shared(n).iS_W(ind,ind) = ehmm.state(k).W.iS_W;
        ehmm.state_shared(n).S_W(ind,ind) = ehmm.state(k).W.S_W;
        ehmm.state_shared(n).Mu_W(ind) = ehmm.state(k).W.Mu_W;
    end
%     if ndim == 1
%         ehmm.state(k).W.S_W = reshape(ehmm.state(k).W.S_W,[1 np np]);
%         ehmm.state(k).W.iS_W = reshape(ehmm.state(k).W.iS_W,[1 np np]);
%     end
end

% setstateoptions;
% K = size(Gamma,2); ndim = size(ehmm.train.S,1); np = size(XX,2);
% 
% Gamma = [Gamma prod(1-Gamma,2) ];
% Gamma = rdiv(Gamma,sum(Gamma,2));  
% 
% for k = 1:K % 1:K+1 for updating baseline as well
%     
%     G = Gamma(:,k);
%     XG = bsxfun(@times, XX, G);
%     gram = XG' * XX;
%     XR = XG' * residuals; 
%     
%     for n = 1:ndim
%         
%         if ~regressed(n), continue; end
%         ndim_n = sum(S(:,n)>0);
%         
%         if ~isempty(lambda)
%             regterm = lambda * eye(np);
%             c = 1;
%         else
%             regterm = [];
%             if ~train.zeromean, regterm = ehmm.state(k).prior.Mean.iS(n); end
%             if ehmm.train.order > 0
%                 alphaterm = ...
%                     repmat( (ehmm.state(k).alpha.Gam_shape ./  ...
%                     ehmm.state(k).alpha.Gam_rate), ndim_n, 1);
%                 if ndim==1
%                     regterm = [regterm; alphaterm(:) ];
%                 else
%                     sigmaterm = repmat(ehmm.state(k).sigma.Gam_shape(S(:,n),n) ./ ...
%                         ehmm.state(k).sigma.Gam_rate(S(:,n),n), length(orders), 1);
%                     regterm = [regterm; sigmaterm .* alphaterm(:) ];
%                 end
%             end
%             c = ehmm.Omega.Gam_shape / ehmm.Omega.Gam_rate(n);
%             regterm = diag(regterm);
%         end
% 
%         iS_W = regterm(Sind(:,n),Sind(:,n)) + Tfactor * c * gram(Sind(:,n),Sind(:,n));
%         iS_W = (iS_W + iS_W') / 2;
%         S_W = inv(iS_W);
%         Mu_W = Tfactor * c * S_W * XR(Sind(:,n),n);
%         
%         ehmm.state(k).W.iS_W(n,:,:) = iS_W;
%         ehmm.state(k).W.S_W(n,:,:) = S_W;
%         ehmm.state(k).W.Mu_W(:,n) = Mu_W;
%         
%         ind = (k-1)*np + (1:np);
%         ehmm.state_shared(n).iS_W(ind,ind) = iS_W;
%         ehmm.state_shared(n).S_W(ind,ind) = S_W;
%         ehmm.state_shared(n).Mu_W(ind) = Mu_W;
%         
%     end
%     
% end

end

