function Gamma = id_Gamma_inference(L,Pi,order)
% inference for independent samples (ignoring time structure)

T = size(L,1) + order; K = length(Pi);
Gamma = zeros(T,K);
Gamma(1+order:end,:) = repmat(Pi,size(L,1),1) .* L(1+order:end,:);
Gamma = rdiv(Gamma,sum(Gamma,2));

end