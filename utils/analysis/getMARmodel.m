function W = getMARmodel(hmm,k)
% Get the MAR coefficients for state k from the estimated model hmm
% This function assumes that order>0
%
% Diego Vidaurre, OHBA, University of Oxford (2016)

if hmm.train.order == 0
    error('The states are not modelled to be MAR distributions (order=0)')
end
model_mean = ~hmm.train.zeromean;

W = hmm.state(k).W.Mu_W(model_mean+1,:);

end