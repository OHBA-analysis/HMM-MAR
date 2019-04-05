function s = getBehAssociation(Gamma,y,T,synch)
% Returns, for each time point, how well the state time courses predict
% the variable y across trials, which must have dimension (trials by 1)
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2018)

if ~all(T(1)==T)
    error('Synchronisity can only be measured if trials have the same length'); 
end 
N = length(T); 
ttrial = size(Gamma,1)/N;
if nargin<4, synch = zeros(1,T(1)); end
K = size(Gamma,2);
Gamma = permute(reshape(Gamma,[ttrial N K]),[2 3 1]);
s = zeros(1,ttrial);
these = ~isnan(y);
y = y(these);
N = length(y);
y = zscore(y);
for t = 1:ttrial
   if synch(t)==1, continue; end
   x = [ones(N,1) Gamma(these,1:K-1,t)];
   beta = (x' * x + 1e-8 * eye(K) ) \ (x' *  y); 
   s(t) = 1 - sum((y - x * beta).^2) / sum(y.^2);
end

end