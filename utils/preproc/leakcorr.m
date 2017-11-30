function [data,M] = leakcorr (data,T,order)
% Correction for volume conduction, when working with source space M/EEG
% If order > 0, it performs Pasqual-Marqui's method (order refers to the MAR order)
%   Pascual-Marqui et al. (2017)
% If order < 1, it performs symmetric orthogonalisation as specified in 
%   Colclough et al. (2015)
% Output M is only defined for Pasqual-Marqui's method
% Needs ROINETS on path
%
% Diego Vidaurre, OHBA, University of Oxford (2017)

if order > 0 
    if isstruct(data)
        [XX,Y] = formautoregr(data.X,T,1:order,order,1);
    else
        [XX,Y] = formautoregr(data,T,1:order,order,1);
    end
    B = XX \ Y;
    eta = Y - XX * B;
    epsilon = ROInets.closest_orthogonal_matrix(eta);
    M = pinv(epsilon) * eta;
    if isstruct(data)
        data.X = data.X * inv(M);
    else
        data = data * inv(M);
    end
else
    if isstruct(data)
        data.X = ROInets.closest_orthogonal_matrix(data.X);
    else
        data = ROInets.closest_orthogonal_matrix(data);
    end
    M = [];
end

end