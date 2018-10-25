function [D] = gauss_kl (mu_q,mu_p,sigma_q,sigma_p)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
%   [D] = gauss_kl (mu_q,mu_p,sigma_q,sigma_p
%
%   computes the divergence 
%                /
%      D(q||p) = | q(x)*log(q(x)/p(x)) dx
%               /
%   between two k-dimensional Gaussian probability
%   densities  given means mu and Covariance Matrices sigam where the
%   Gaussian pdf is given by  
%
%              1                                     T       -1
%   p(x)= ------------------------- exp (-0.5  (x-mu)   Sigma    (x-mu)  )
%         (2*pi)^(d/2) |Sigma|^0.5
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if nargin<4
  error('Incorrect number of input arguments');
end

if length(mu_q)~=length(mu_p)
  error('Distributions must have equal dimensions (Means dimension)');
end
mu_q=mu_q(:);
mu_p=mu_p(:);

if size(sigma_q)~=size(sigma_p)
  error('Distributions must have equal dimensions (Covariance dimension)');
end

%DSq=det(sigma_q);
%DSp=det(sigma_p);
K=size(sigma_q,1);
isigmap = inv(sigma_p);

D=logdet(sigma_p) - logdet(sigma_q) -K+trace(isigmap*sigma_q)+(mu_q-mu_p)'*isigmap*(mu_q-mu_p);
D=D*0.5;
