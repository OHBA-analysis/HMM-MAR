
function [D] = gamma_kl (shape_q,shape_p,rate_q,rate_p)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   [D] = gamma_kl (shape_q,shape_p,rate_q,rate_p)
%
%   computes the divergence 
%                /
%      D(q||p) = | q(x)*log(q(x)/p(x)) dx
%               /
%   between two gamma propability densities
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

scale_p = 1 / rate_p;
scale_q = 1 / rate_q;

if nargin<4,
  error('Incorrect number of input arguments');
end;

D =  (shape_q - 1) * psi(shape_q) - log(scale_q) - shape_q - gammaln(shape_q) + ...
    gammaln(shape_p) + shape_p * log(scale_p) - (shape_p - 1) * (psi(shape_q) + log(scale_q)) + ...
    scale_q * shape_q / scale_p;

return;
