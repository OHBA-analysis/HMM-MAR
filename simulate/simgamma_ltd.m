function Gamma = simgamma_ltd(T,P,Pi,rate,L,refractory_period,grouping)
%
% Simulate state time courses with longer-than-markov time dependencies
%
% INPUTS:
%
% T                     Number of time points for each time series
% P                     Transition probability matrix (K by K)
% Pi                    Initial probabilities (K by 1)
% rate                  The weights that model the contribution of the
%                       latest L points are modelled by a Gamma
%                       distribution, whose shape is 1 - and rate is
%                       specified here
% L                     The length of the history (in number of time points)
%                       that influences the state at time t
% refractory_period     to prevent bursts of quick changes, refractory_period
%                       can be set so that, after a change, you cannot change again 
%                       after 'refractory_period' number of iterations
%
% OUTPUTS
%
% Gamma         simulated  p(state | data)
%
% Author: Diego Vidaurre, OHBA, University of Oxford

N = length(T); K = length(Pi);

if nargin<4, rate = 2; end
if nargin<5, L = 10; end 
if nargin<6, refractory_period = 2; end 
if nargin<7, grouping = []; end

Gamma = zeros(sum(T),K);
weights = gampdf(0:L-1,1,rate)'; 
weights = weights / sum(weights);
weights = weights(end:-1:1);

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
        Gammai(1,:) = mnrnd(1,Pin);
    end
    last_ch = Inf; 
    for t=2:L 
        if last_ch < refractory_period
            Gammai(t,:) = Gammai(t-1,:); 
            last_ch = last_ch + 1;
        else
            if t==2
                g = repmat(weights(end-t+2:end,1),1,K) .* Gammai(1:t-1,:);
            else
                g = sum(repmat(weights(end-t+2:end,1),1,K) .* Gammai(1:t-1,:));
            end
            g = g / sum(g);
            Gammai(t,:) = mnrnd(1,g*Pn);
            if any(Gammai(t,:)~=Gammai(t-1,:)), last_ch = 1; end
            Gammai(t,:) = Gammai(t,:) / sum(Gammai(t,:));   
        end
    end
    for t=L+1:T(n)
        if last_ch < refractory_period
            Gammai(t,:) = Gammai(t-1,:);
            last_ch = last_ch + 1;
        else
            g = sum(repmat(weights,1,K) .* Gammai(t-L:t-1,:));
            Gammai(t,:) = mnrnd(1,g*Pn);
            if any(Gammai(t,:)~=Gammai(t-1,:)), last_ch = 1; end
        end
    end
    t = (1:T(n)) + sum(T(1:n-1));
    Gamma(t,:) = Gammai;
end

end

