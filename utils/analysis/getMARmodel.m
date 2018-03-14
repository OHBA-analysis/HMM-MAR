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

if isfield(hmm.train,'A')
    A = hmm.train.A;
    order = size(W,1) / size(W,2);
    W0 = W;
    W = zeros(order*size(A,2),size(A,2));
    for i = 1:order
        ii = (1:size(W0,2)) + (i-1) * size(W0,2);
        jj = (1:size(W,2)) + (i-1) * size(W,2);
        W(jj,:) = A * W0(ii,:) * A';
    end
end

end