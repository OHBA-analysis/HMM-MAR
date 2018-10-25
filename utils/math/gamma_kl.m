function D = gamma_kl (shape_q,shape_p,rate_q,rate_p)
% Computes D(q||p) = | q(x)*log(q(x)/p(x)) dx
% between two Gamma probability densities
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2016)

if nargin<4
  error('Incorrect number of input arguments');
end

% scale_p = 1 / rate_p;
% scale_q = 1 / rate_q;% D =  (shape_q - 1) * psi(shape_q) - log(scale_p) - shape_q - gammaln(shape_q) + ...
%     gammaln(shape_p) + shape_p * log(scale_p) - (shape_p - 1) * (psi(shape_q) + log(scale_q)) + ...
%     scale_q * shape_q / scale_p;
% 
% end

n = max([size(rate_q,2) size(shape_q,2)]);

if size(shape_q,2) == 1, shape_q = shape_q*ones(1,n); end
if size(rate_q,2) == 1, rate_q = rate_q*ones(1,n); end
shape_p = shape_p*ones(1,n); rate_p = rate_p*ones(1,n);

D = sum( shape_q.*log(rate_q)-gammaln(shape_q) ...
         -shape_p.*log(rate_p)+gammaln(shape_p) ...
	 +(shape_q-shape_p).*(psi(shape_q)-log(rate_q)) ...
	 -(rate_q-rate_p).*shape_q./rate_q ,2);
 
end
