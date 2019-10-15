function [hmm] = updatePsi(hmm,Gamma,X,Y)
% Computes the parameter psi for tuning the approximation to the
% logistic function.

[T,ndim]=size(X);
% Gamma = hmm.Gamma; % warning: this is not guaranteed to be the updated Gamma!
K = hmm.train.K;
if length(unique(Y(:)))<3
    binomial=true;
else 
    binomial=false;
end
for iY=1:hmm.train.logisticYdim
    WW=cell(K,1);
    sum1=zeros(T,ndim);
    for i=1:K
        WW{i}=hmm.state(i).W.Mu_W(1:ndim,ndim+iY)*hmm.state(i).W.Mu_W(1:ndim,ndim+iY)' + ...
                    squeeze(hmm.state(i).W.S_W(ndim+iY,1:ndim,1:ndim));
        if binomial            
            sum1=sum1 + (repmat(Gamma(:,i),1,ndim).*X)*WW{i};
            vp=1:T;
        else
            vp= Y(:,iY)~=0;
            sum1(vp,:)=sum1(vp,:) + (repmat(Gamma(vp,i),1,ndim).*X(vp,:))*WW{i};
        end
    end     
    hmm.psi(1:T,iY) = sqrt(sum(sum1 .*X,2));
    
    if ~binomial
        hmm.psi(~vp,iY)=NaN;  % these points are invalid; put as NaN's to avoid error propogation 
    end
    
    %older, slower loop code:
%     for i=1:K
%         WW{i}=hmm.state(i).W.Mu_W(1:ndim,ndim+iY)*hmm.state(i).W.Mu_W(1:ndim,ndim+iY)' + ...
%                     squeeze(hmm.state(i).W.S_W(ndim+iY,1:ndim,1:ndim));
%     end
%     if ~isfield(hmm,'psi')
%         hmm.psi=zeros(T,1);
%         for t=1:T
%             gamWW=zeros(ndim_n,ndim_n,K);
%             for i=1:K
% %                         gamWW(:,:,i) = Gamma(t,i)* ...
% %                             (hmm.state(i).W.Mu_W(Sind(:,n),n)*hmm.state(i).W.Mu_W(Sind(:,n),n)' + ...
% %                             squeeze(hmm.state(i).W.S_W(n,S(:,n),S(:,n))));
%                   gamWW(:,:,i) = Gamma(t,i)*WW{i};
%             end
%             hmm.psi(t)=sqrt(X(t,:) * sum(gamWW,3) * X(t,:)');
%         end
%     end
    
    
end

end