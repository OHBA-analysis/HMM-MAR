function Gamma = simgamma(T,P,Pi,nrep)
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

N = length(T); K = length(Pi);
if nargin<4, nrep = 1; end

Gamma = zeros(sum(T),K);

for in=1:N
    Gammai = zeros(T(in),K);
    if any(Pi==1)
        Gammai(1,Pi==1) = 1;
    else
        Gammai(1,:) = mnrnd(nrep,Pi);
        if nrep>1, Gammai(1,:) = Gammai(1,:) / sum(Gammai(1,:)); end
    end
    for t=2:T(in)
        Gammai(t,:) = mnrnd(nrep,Gammai(t-1,:) * P);
        if nrep>1,  Gammai(t,:) = Gammai(t,:) / sum(Gammai(t,:)); end
    end
    t = (1:T(in)) + sum(T(1:in-1));
    Gamma(t,:) = Gammai;
end

end

