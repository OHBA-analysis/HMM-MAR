function Entr = GammaEntropy(Gamma,Xi,T,order)
% Entropy of the state time courses
Entr = 0; K = size(Gamma,2);
for tr = 1:length(T)
    t = sum(T(1:tr-1)) - (tr-1)*order + 1;
    Gamma_nz = Gamma(t,:); 
    Gamma_nz(Gamma_nz==0) = realmin;
    if any(isinf(log(Gamma_nz(:)))), Gamma_nz(Gamma_nz==0) = eps; end
    Entr = Entr - sum(Gamma_nz.*log(Gamma_nz));
    if ~isempty(Xi)
        t = (sum(T(1:tr-1)) - (tr-1)*(order+1) + 1) : ((sum(T(1:tr)) - tr*(order+1)));
        Xi_nz = Xi(t,:,:);
        Xi_nz(Xi_nz==0) = realmin;
        if any(isinf(log(Xi_nz(:)))), Xi_nz(Xi_nz==0) = eps; end
        Psi = zeros(size(Xi_nz));                    % P(S_t|S_t-1)
        for k = 1:K
            sXi = sum(permute(Xi_nz(:,k,:),[1 3 2]),2);
            Psi(:,k,:) = Xi_nz(:,k,:)./repmat(sXi,[1 1 K]);
        end
        Psi(Psi==0) = realmin;
        if any(isinf(log(Psi(:)))), Psi(Psi==0) = eps; end
        Entr = Entr - sum(Xi_nz(:).*log(Psi(:)));    % entropy of hidden states
    end
end
if isnan(Entr(:)) 
    error(['Error computing entropy of the state time courses  - ' ...
        'Out of precision?'])     
end
end
