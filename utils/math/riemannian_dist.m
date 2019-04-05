function d = riemannian_dist(C1,C2)
% computes the riemannian distance between two positive definite matrices,
% C1 and C2
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2019)

d = sqrt(sum(log(eig(C1,C2)).^2));

end