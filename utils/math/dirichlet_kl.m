function D = dirichlet_kl(alpha_q,alpha_p)
% Computes D(q||p) = | q(x)*log(q(x)/p(x)) dx
% between two k-dimensional Dirichlet probability densities, with pdf
%          
%         Gamma(sum_{i=1}^k alpha_i)     alpha_1-1        alpha_k-1 
%   p(x)= --------------------------- x_1         .... x_k
%         prod_{i=1}^k Gamma(alpha_i)
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2016)            

ind = (alpha_q>0) & (alpha_p>0);
alpha_q = alpha_q(ind);
alpha_p = alpha_p(ind);

if nargin<2
  error('Incorrect number of input arguments');
end

if length(alpha_q)~=length(alpha_p)
  error('Distributions must have equal dimensions');
end

D = gammaln(sum(alpha_q)) - gammaln(sum(alpha_p)) - ...
    sum(gammaln(alpha_q)) + sum(gammaln(alpha_p)) + ...
    (alpha_q - alpha_p) * (psi(alpha_q) - psi(sum(alpha_q)))';

end


