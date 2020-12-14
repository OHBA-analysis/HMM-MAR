function C = array_multiplication(A,B,G)
% A is (N X p), B is (N X p X p), we want: C(t,:) = A(t,:) x B(t,:,:)
% if G is specified, C(t,:) = G(t) x (A(t,:) x B(t,:,:))
% Author: Diego Vidaurre, OHBA, University of Oxford
if nargin < 3, G = []; end
[N,p] = size(A);
C = reshape(repmat(A,1,p) .* reshape(B,N,p*p), N, p, p);
C = permute(sum(C,2),[1 3 2]);
if ~isempty(G)
   C = bsxfun(@times,C,G); 
end
end