function Gamma = vpath_to_stc(vpath,K)
% changes a viterbi path (T x 1) to a state time course (T x K) format

if size(vpath,2)>1, error('Viterbi paths have one single column'); end
if nargin<2, K = max(vpath); end
Gamma = zeros(numel(vpath),K,'double');
for k=1:K, Gamma(vpath==k,k) = 1; end        

end
