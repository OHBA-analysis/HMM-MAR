function LifeTimes = getStateLifeTimes (Gamma,T,threshold,threshold_Gamma)
% Computes the state life times for the state time courses  
%
% Gamma can be the probabilistic state time courses (time by states),
%   which can contain the probability of all states or a subset of them,
%   or the Viterbi path (time by 1). 
% In the first case, threshold_Gamma is used to define when a state is active. 
% The parameter threshold is used to discard visits that are too
%   short (deemed to be spurious); these are discarded when they are shorter than 
%   'threshold' time points
% LifeTimes is then a cell (subjects by states), where each element is a
%   vector of life times
%
% Diego Vidaurre, OHBA, University of Oxford (2016)

is_vpath = (size(Gamma,2)==1 && all(rem(Gamma,1)==0));
if nargin<3, threshold = 0; end
if nargin<4, threshold_Gamma = (2/3); end
if iscell(T)
    for i = 1:length(T)
        if size(T{i},1)==1, T{i} = T{i}'; end
    end
    T = cell2mat(T);
end
N = length(T);
if is_vpath % viterbi path
    vpath = Gamma; 
    K = length(unique(vpath));
    Gamma = zeros(length(vpath),K);
    for k = 1:K
       Gamma(vpath==k,k) = 1;   
    end
else
    K = size(Gamma,2); 
    Gamma = Gamma > threshold_Gamma;
end

LifeTimes = cell(N,K);
order = (sum(T)-size(Gamma,1))/length(T);

for j=1:N
    t0 = sum(T(1:j-1)) - (j-1)*order;
    ind = (1:T(j)-order) + t0;
    for k=1:K
        LifeTimes{j,k} = aux_k(Gamma(ind,k),threshold);
    end
end

if length(LifeTimes)==1, LifeTimes = LifeTimes{1}; end

end


function LifeTimes = aux_k(g,threshold)

LifeTimes = [];

while ~isempty(g)
    
    t = find(g==1,1);
    if isempty(t), break; end
    tend = find(g(t+1:end)==0,1);
    
    if isempty(tend) % end of trial
        if (length(g)-t+1)>threshold
            LifeTimes = [LifeTimes (length(g)-t+1)];
        else
            g(t:end) = 0; % too short visit to consider
        end
        break;
    end
    
    if tend<threshold % too short visit, resetting g and trying again
        g(t:t+tend-1) = 0;
        continue;
    end
    
    LifeTimes = [LifeTimes tend];
    tnext = t + tend;
    g = g(tnext:end);
    
end

end
