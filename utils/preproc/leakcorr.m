function [data,M] = leakcorr (data,T,order)
% Correction for volume conduction, when working with source space M/EEG
% If order > 0, it performs Pasqual-Marqui's method (order refers to the MAR order)
%   Pascual-Marqui et al. (2017)
% If order < 1, it performs symmetric orthogonalisation as specified in 
%   Colclough et al. (2015)
% Output M is only defined for Pasqual-Marqui's method
% Needs ROINETS on path

if order > 0 
    [XX,Y] = formautoregr(data,T,1:order,order,1);
    B = XX \ Y;
    eta = Y - XX * B;
    epsilon = ROInets.closest_orthogonal_matrix(eta);
    M = pinv(epsilon) * eta;
    data = data * inv(M);
else
    data = ROInets.closest_orthogonal_matrix(data);
    M = [];
end

end