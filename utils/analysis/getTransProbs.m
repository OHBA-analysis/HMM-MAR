function TP = getTransProbs (hmm)
% Returns the transition probabilies from any state to any other state,
% without considering the persistence probabilities (i.e. the probability
% to remain in the same state)
%
% Author: Diego Vidaurre, University of Oxford (2017)

TP = hmm.P;
[K,~,Q] = size(TP);
for j = 1:Q
   TPj = TP(:,:,j);
   TPj(eye(K)==1) = 0;
   for k = 1:K
      TPj(k,:) = TPj(k,:) / sum(TPj(k,:)); 
   end
   TP(:,:,j) = TPj; 
end

end