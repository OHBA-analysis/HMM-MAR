function FO = fractionalOccupancy (Gamma,T,is_vpath)
% computes de fractional occupancy for each trial

if nargin<3, is_vpath = (size(Gamma,2)==1 && all(rem(Gamma,1)==0)); end
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

FO = zeros(N,K);
order = (sum(T)-size(Gamma,1))/length(T);

for j=1:N
    t0 = sum(T(1:j-1)) - (j-1)*order;
    ind = (1:T(j)-order) + t0;
    FO(j,:) = mean(Gamma(ind,:));
end

if length(FO)==1, FO = FO{1}; end

end

