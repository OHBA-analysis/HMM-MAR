function [Gamma,scale] = id_Gamma_inference(L,Pi,order)
% inference for independent samples (ignoring time structure)
% scale is the likelihood

L(L<realmin) = realmin;
Gamma = repmat(Pi,size(L,1),1) .* L;
Gamma = Gamma(1+order:end,:);
scale = sum(Gamma,2);	
Gamma = rdiv(Gamma,scale);

end