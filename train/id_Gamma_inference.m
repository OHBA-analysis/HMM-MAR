function Gamma = id_Gamma_inference(L,Pi,order)
% inference for independent samples (ignoring time structure)

Gamma = zeros(T,K);
Gamma(1+order,:) = repmat(Pi,size(L,1),1) .* L(1+order,:);
Gamma = rdiv(Gamma,sum(Gamma,2));

end