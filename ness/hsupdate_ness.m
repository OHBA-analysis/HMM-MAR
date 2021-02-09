function ness = hsupdate_ness(Xi,Gamma,T,ness)
%
% updates hidden state parameters of an HMM
%
% INPUT:
%
% Xi     probability of past and future state cond. on data
% Gamma  probability of current state cond. on data
% T      length of observation sequences
% hmm    single hmm data structure
%
% OUTPUT
% hmm    single hmm data structure with updated state model probs.
%
% Author: Diego Vidaurre, OHBA, University of Oxford

K = ness.K;

if isempty(Xi)  % non-exact estimation
    Xi = approximateXi(Gamma,T,ness);
end
if length(size(Xi))==4
    Xi = permute(sum(Xi),[2 3 4 1]);
end

% transitions
for k = 1:K
    ness.state(k).Dir2d_alpha = permute(Xi(k,:,:),[2 3 1]) ...
        + ness.state(k).prior.Dir2d_alpha;
    ness.state(k).P = zeros(2);
    for j = 1:2
        PsiSum = psi(sum(ness.state(k).Dir2d_alpha(j,:)));
        for j2 = 1:2
            ness.state(k).P(j,j2) = ...
                exp(psi(ness.state(k).Dir2d_alpha(j,j2))-PsiSum);
        end
        ness.state(k).P(j,:) = ness.state(k).P(j,:) ./ sum(ness.state(k).P(j,:));
    end
end

% initial state is always OFF for all chains
end