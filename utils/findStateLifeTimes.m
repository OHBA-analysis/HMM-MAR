function [LifeTimes,Gamma] = findStateLifeTimes (Gamma,T,threshold)
% find the state life times for the state time courses of one state
% Gamma needs to be (time by 1), and the sum of T be equal to size(Gamma,1)

if nargin<3, threshold = 0 ; end
K = size(Gamma,2); N = length(T);
Gamma = Gamma > (2/3);

LifeTimes = cell(K,1);

for j=1:N
    t0 = sum(T(1:j-1));
    ind = (1:T(j)) + t0;
    for k=1:K
        [lfk,g] = rec_k(Gamma(ind,k),threshold);
        LifeTimes{k} = [LifeTimes{k} lfk];
        Gamma(ind,k) = g;
    end
end

if length(LifeTimes)==1, LifeTimes = LifeTimes{1}; end

end


function [LifeTimes,g] = rec_k(g,threshold)

LifeTimes = [];

if isempty(g), return; end
t = find(g==1,1);
if isempty(t), return; end
tend = find(g(t+1:end)==0,1);

if isempty(tend) % end of trial
    if (length(g)-t+1)>threshold
        LifeTimes = (length(g)-t+1);
    else
        g(t:end) = 0; % too short visit to consider
    end
    return;
end

if tend<threshold % too short visit, resetting g and trying again
   g(t:t+tend-1) = 0;  
   [LifeTimes,g] = rec_k(g,threshold);
   return;
end

LifeTimes = tend;
tnext = t + tend; 
[ltnext,gnext] = rec_k(g(tnext:end),threshold);
LifeTimes = [LifeTimes ltnext];
g(tnext:end) = gnext;

end
