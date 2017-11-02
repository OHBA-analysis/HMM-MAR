function switchingRate =  getSwitchingRate(Gamma,T)
% Computes the state switching rate for each session/subject
%
% Gamma can be the probabilistic state time courses (time by states),
%   which can contain the probability of all states or a subset of them,
%   or the Viterbi path (time by 1). 
% switchingRate is a vector (sessions/subjects by 1) 
%
% Diego Vidaurre, OHBA, University of Oxford (2017)

is_vpath = (size(Gamma,2)==1 && all(rem(Gamma,1)==0)); % is a viterbi path?

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
end

switchingRate = zeros(N,1);

for n=1:N
    t = sum(T(1:n-1)) + (1:T(n));
    switchingRate(n) = mean(sum(abs(diff(Gamma(t,:))),2));
end

end