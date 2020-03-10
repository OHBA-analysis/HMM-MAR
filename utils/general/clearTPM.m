function hmm = clearTPM(hmm)
% Removes the information about the state transition probabilities (hmm.P)
% and the initial probabilities (hmm.Pi).
% This is useful if the model is to be applied to a different data set
% where the underlying system may be different
%
% Author: Diego Vidaurre (2020) University of Oxford

K = length(hmm.Pi);
hmm.Pi(:) = 1/K;
hmm.Dir_alpha(:) = mean(hmm.Dir_alpha);
d = mean(diag(hmm.P));
hmm.P(:) = (1 - d) / (K - 1);
hmm.P(eye(K)==1) = d; 
d1 = mean(diag(hmm.Dir2d_alpha));
d2 = mean(hmm.Dir2d_alpha(eye(K)~=1));
hmm.Dir2d_alpha(:) = d2;
hmm.Dir2d_alpha(eye(K)==1) = d1;

end