function Intervals = findStateIntervalTimes (Gamma,T,threshold,is_vpath)
% find the interval times for the state time courses of one state
% Gamma needs to be (time by 1), and the sum of T be equal to size(Gamma,1)

if nargin<3, threshold = 0 ; end
if nargin<4, is_vpath = (size(Gamma,2)==1 && all(rem(Gamma,1)==0)); end
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

Intervals = cell(K,1);
order = (sum(T)-size(Gamma,1))/length(T);

for j=1:N
    t0 = sum(T(1:j-1)) - (j-1)*order;
    ind = (1:T(j)-order) + t0;
    for k=1:K
        t = find(Gamma(ind,k)==1,1);
        if isempty(t), continue; end
        lfk = aux_k(Gamma(ind(t:end),k),threshold);
        Intervals{k} = [Intervals{k} lfk];
        %Gamma(ind,k) = g;
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
