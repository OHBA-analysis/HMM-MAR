function Gamma = simgamma(T,P,Pi,nrep,grouping)
%
% Simulate data from the HMM-MAR
%
% INPUTS:
%
% T                     Number of time points for each time series
% P                     Transition probability matrix
% Pi                    Initial probabilities
% nrep                  no. repetitions of Gamma(t), from which we take the average
%
% OUTPUTS
%
% Gamma         simulated  p(state | data)
%
% Author: Diego Vidaurre, OHBA, University of Oxford

if nargin<5, grouping = []; end
rng('shuffle') % make this "truly" random

N = length(T); K = length(Pi);
if nargin<4, nrep = 1; end

Gamma = zeros(sum(T),K);

for n = 1:N
    if ~isempty(grouping)
        i = grouping(n);
        Pn = P(:,:,i); Pin = Pi(:,i)';
    else
        Pn = P; Pin = Pi;
    end
    Gammai = zeros(T(n),K);
    if any(Pin==1)
        Gammai(1,Pin==1) = 1;
    else
        Gammai(1,:) = mnrnd(nrep,Pin);
        if nrep>1, Gammai(1,:) = Gammai(1,:) / sum(Gammai(1,:)); end
    end
    for t=2:T(n)
        Gammai(t,:) = mnrnd(nrep,Gammai(t-1,:) * Pn);
        if nrep>1,  Gammai(t,:) = Gammai(t,:) / sum(Gammai(t,:)); end
    end
    t = (1:T(n)) + sum(T(1:n-1));
    Gamma(t,:) = Gammai;
end

end

