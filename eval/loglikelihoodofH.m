function [Q_h] = loglikelihoodofH(Y,X,hmm)
% function calculates the log likelihood of our approximate likelihood
% function h(Y|w,gamma,xi), which approximates the likelihood P(Y|w,gamma)
% for bayesian weighted logistic regression models.

[T,dimY]=size(Y);
[~,dimX]=size(X);

% if dimX ~= size(hmm.state(1).w_mu,1)
%     ME=MException(loglikelihoodofH:xdimension, ...
%         'Error: X dimension inputted does not match W dimension');
%     throw ME;
% end
if any(~(Y(:)==1 | Y(:)==-1 | Y(:)==0))
    ME = MException(loglikelihoodofH:Yformaterror,'Error; Y not in correct format for binary inference')
    throw ME;
end

if ~isfield(hmm,'psi')
    MException(loglikelihoodofH:nopsirec,'Error; No record of PSI in hmm structure')
    throw ME;
end

Sind=hmm.train.S==1;
%n=dimX+1;
outerWprod = @(hmm,i,n,Xdim) squeeze(hmm.state(i).W.S_W(n,1:Xdim,1:Xdim)) + ...
         hmm.state(i).W.Mu_W(1:Xdim,Xdim+n)*hmm.state(i).W.Mu_W(1:Xdim,Xdim+n)';
Q_h=zeros(T,dimY);
for iY=1:dimY
    vp = Y(:,iY)~=0;
    n=dimX+iY;
    % calculation broken down into component parts for ease of reference:
    comp1 = log (log_sigmoid(hmm.psi(vp,iY)));

    comp2 = zeros(sum(vp),1);
    for i=1:length(hmm.state)
        comp2= comp2 + (repmat(hmm.Gamma(vp,i),1,dimX) .* X(vp,:)) * hmm.state(i).W.Mu_W(1:dimX,dimX+iY);
    end
    comp2 = (Y(vp,iY).*comp2 - hmm.psi(vp,iY))/2;

    lambdafunc = @(xi_in) ((2*xi_in).^-1).*(log_sigmoid(xi_in)-0.5);
    comp3 = -(hmm.psi(vp,iY).^2);
    for i=1:length(hmm.state)
        comp3 = comp3 + sum((repmat(hmm.Gamma(vp,i),1,dimX) .* X(vp,:)) * outerWprod(hmm,i,iY,dimX) .* X(vp,:),2);
        %(params.state(i).w_mu*params.state(i).w_mu') * X';
    end
    comp3 = lambdafunc(hmm.psi(vp,iY)).*comp3;

    Q_h(vp,iY) = comp1 + comp2 - comp3;
end
% if any(~(Y(:)==0))
%     % remove invalid computations 
%     mask = 1-[Y==0];
%     Q_h = Q_h.*mask;
% end
Q_h=sum(Q_h,2);
end

% function WW = outerWprod(hmm,i,n)
% WW=zeros(n-1,n-1);
% %dimY=hmm.train.logisticYdim;
% % for nY=1%:dimY
% % %     WW = WW + (squeeze(hmm.state(i).W.S_W(nY+n-1,1:n-1,1:n-1)) + ...
% % %         hmm.state(i).W.Mu_W(1:n-1,n:n+dimY-1)*hmm.state(i).W.Mu_W(1:n-1,n:n+dimY-1)');
% % 
% % end
% WW = squeeze(hmm.state(i).W.S_W(nY+n-1,1:n-1,1:n-1)) + ...
%          hmm.state(i).W.Mu_W(1:n-1,n:n+dimY-1)*hmm.state(i).W.Mu_W(1:n-1,n:n+dimY-1)');
% end