function Intervals = findStateIntervalTimes (Gamma,T,threshold,threshold_Gamma)
% find the interval times for the state time courses of one state
% Gamma can be the probabilistic state time courses (time by states),
%   which can contain the probability of all states or a subset of them,
%   or the Viterbi path (time by 1). 
% In the first case, threshold_Gamma is used to define when a state is active. 
% The parameter threshold is used to discard intervals that are too
%   short (deemed to be spurious); these are discarded when they are shorter than 
%   'threshold' time points
% Intervals is then a cell (subjects by states), where each element is a
%   vector of interval times
%
% Diego Vidaurre, OHBA, University of Oxford (2016)

is_vpath = (size(Gamma,2)==1 && all(rem(Gamma,1)==0)); % is a viterbi path?
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

Intervals = cell(N,K);
order = (sum(T)-size(Gamma,1))/length(T);

for j=1:N
    t0 = sum(T(1:j-1)) - (j-1)*order;
    ind = (1:T(j)-order) + t0;
    for k=1:K
        t = find(Gamma(ind,k)==1,1);
        if isempty(t), continue; end
        Intervals{j,k} = aux_k(Gamma(ind(t:end),k),threshold);
    end
end

if length(Intervals)==1, Intervals = Intervals{1}; end

end



function Intervals = aux_k(g,threshold)
% g starts with 1

Intervals = [];

while ~isempty(g)
    
    t = find(g==0,1);
    if isempty(t), break; end % end of trial - trial finishes active
    tend = find(g(t+1:end)==1,1); % end of interval
    if isempty(tend), break; end % end of trial - no more visits
    
    if tend<threshold % too short interval, resetting g and trying again
        g(t:t+tend-1) = 1;
        continue
    end
    
    Intervals = [Intervals tend];
    tnext = t + tend;
    g = g(tnext:end);
end


end
