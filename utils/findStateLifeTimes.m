function LifeTimes = findStateLifeTimes (Gamma,T,threshold,is_vpath)
% find the state life times for the state time courses of one state
% Gamma needs to be (time by 1), and the sum of T be equal to size(Gamma,1)
% is_vpath (optional) indicates whether Gamma is viterbi path or a
% probabilistic assignment

if nargin<3, threshold = 0 ; end
if nargin<4, is_vpath = size(Gamma,2)==1; end
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
    Gamma = Gamma > (2/3);
end

LifeTimes = cell(K,1);
order = (sum(T)-size(Gamma,1))/length(T);

for j=1:N
    t0 = sum(T(1:j-1)) - (j-1)*order;
    ind = (1:T(j)-order) + t0;
    for k=1:K
        lfk = aux_k(Gamma(ind,k),threshold);
        LifeTimes{k} = [LifeTimes{k} lfk];
        %Gamma(ind,k) = g;
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
