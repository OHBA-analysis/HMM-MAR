function D = beta_kl (a_q,a_p,b_q,b_p)
% Computes D(q||p) = | q(x)*log(q(x)/p(x)) dx
% between two Beta probability densities
%
% Author: Cam Higgins, OHBA, University of Oxford (2019)
if nargin<4
  error('Incorrect number of input arguments');
end

term1 = (a_q - a_p).*psi(a_q) + (b_q - b_p) .* psi(b_q) + ...
    (a_p - a_q +b_p - b_q).*psi(a_q + b_q);
%term2 = log( beta(a_p,b_p) ./ beta(a_q,b_q));
term2 = gammaln(a_p)+gammaln(b_p)-gammaln(a_p+b_p)+gammaln(a_q+b_q)-gammaln(a_q)-gammaln(b_q);
D = sum(term1 + term2);

end