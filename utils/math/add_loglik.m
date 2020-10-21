function LL = add_loglik(L,Gamma,additiveHMM)
% Generalises the sum of loglikelihoods for factorial/additive HMM
if additiveHMM
    Gamma = [Gamma sum(1 - Gamma,2) ];
    %Gamma = rdiv(Gamma,sum(Gamma,2));
end
LL = sum(log(sum(L .* Gamma, 2))); 
end
