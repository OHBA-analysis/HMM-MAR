function Entr = GammaEntropy_ness(Gamma,Xi,T)
% Entropy of the state time courses
Entr = 0; K = size(Gamma,2); order = (sum(T)-size(Gamma,1))/length(T);
% for k = 1:K
%     for tr = 1:length(T)
%         t = (2:T(tr)-order) + sum(T(1:tr-1)) - (tr-1)*order; % start in 2
%         Gamma_nz = [ Gamma(t,k) (1-Gamma(t,k)) ];
%         Gamma_nz(Gamma_nz==0) = realmin;
%         if any(isinf(log(Gamma_nz(:)))), Gamma_nz(Gamma_nz==0) = eps; end
%         Entr = Entr + sum(Gamma_nz(:).*log(Gamma_nz(:)));
%         t = (2:T(tr)-order-1) + sum(T(1:tr-1)) - (tr-1)*(order+1);
%         Xi_nz = Xi(t,k,:,:);
%         Xi_nz(Xi_nz==0) = realmin;
%         if any(isinf(log(Xi_nz(:)))), Xi_nz(Xi_nz==0) = eps; end
%         Entr = Entr - sum(Xi_nz(:).*log(Xi_nz(:)));        
%     end
% end

for k = 1:K
    Gk = [Gamma(:,k) (1-Gamma(:,k))];
    Xik = permute(Xi(:,k,:,:),[1 3 4 2]);
    Entr = Entr + GammaEntropy(Gk,Xik,T,order);
end

if isnan(Entr(:)) 
    error(['Error computing entropy of the state time courses  - ' ...
        'Out of precision?'])     
end
end
