function FO = getFractionalOccupancy (Gamma,T,dim)
% computes de fractional occupancy  
% - across trials if dim==1 (i.e. FO is time by no.states)
% - across time if dim==2 (i.e. FO is no.trials by no.states)
% (default, dim=2)
% Note: this can be applied to the data as well, if you want to look at the
% evoked response.
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2016)

if nargin<3, dim = 2; end
is_vpath = (size(Gamma,2)==1 && all(rem(Gamma,1)==0)); 
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
    %Gamma = Gamma > (2/3);
end

order = (sum(T)-size(Gamma,1))/length(T);

if dim == 2
    FO = zeros(N,K);
    for j=1:N
        t0 = sum(T(1:j-1)) - (j-1)*order;
        ind = (1:T(j)-order) + t0;
        FO(j,:) = mean(Gamma(ind,:));
    end
else
    if any(T~=T(1)) 
        error('All trials must have the same length if dim==1'); 
    end
    Gamma = reshape(Gamma,[T(1)-order,length(T),K]);
    FO = squeeze(mean(Gamma,2));
end

if length(FO)==1, FO = FO{1}; end

end

