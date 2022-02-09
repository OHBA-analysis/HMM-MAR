function ehmm = hsupdate_ehmm(Xi,Gamma,T,ehmm)
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

K = ehmm.K;

if isempty(Xi)  % non-exact estimation
    order = (sum(T)-size(Gamma,1)) / length(T); 
    Xi = zeros(sum(T-1-order),K,2,2);
    for j = 1:length(T)
        indG = (1:(T(j)-order)) + sum(T(1:j-1)) - (j-1)*order;
        indXi =  (1:(T(j)-order-1)) + sum(T(1:j-1)) - (j-1)*(order+1);
        for k = 1:K
            g = [Gamma(indG,k) (1-Gamma(indG,k))];
            for t = 1:length(indXi)
                xi = g(t,:)' * g(t+1,:);
                xi = xi / sum(xi(:));
                Xi(indXi(t),k,:,:) = xi;
            end
        end
    end
end
if length(size(Xi))==4
    Xi = permute(sum(Xi),[2 3 4 1]);
end

% transitions
for k = 1:K
    ehmm.state(k).Dir2d_alpha = permute(Xi(k,:,:),[2 3 1]) ...
        + ehmm.state(k).prior.Dir2d_alpha;
    ehmm.state(k).P = zeros(2);
    for j = 1:2
        PsiSum = psi(sum(ehmm.state(k).Dir2d_alpha(j,:)));
        for j2 = 1:2
            ehmm.state(k).P(j,j2) = ...
                exp(psi(ehmm.state(k).Dir2d_alpha(j,j2))-PsiSum);
        end
        ehmm.state(k).P(j,:) = ehmm.state(k).P(j,:) ./ sum(ehmm.state(k).P(j,:));
    end
end

% initial state is always OFF for all chains
end