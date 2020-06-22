function Gamma = id_Gamma_inference(L,Pi,order)
% inference for independent samples (ignoring time structure)

L(L<realmin) = realmin;
Gamma = repmat(Pi,size(L,1),1) .* L;
Gamma = Gamma(1+order:end,:);
Gamma = rdiv(Gamma,sum(Gamma,2));

end