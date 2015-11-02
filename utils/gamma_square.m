function [M_Gamma2] = gamma_square (Gamma)

M_Gamma2 = zeros(size(Gamma)); 
for k=1:size(Gamma,2)
    M_Gamma2(:,k) = Gamma(:,k).^2 + Gamma(:,k).*(1-Gamma(:,k));
end;