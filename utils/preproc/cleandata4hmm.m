function [data,T] = cleandata4hmm (data,T)

if iscell(T) || iscell(data)
    error('X and T are expected to be matrices here. Run subject by subject.')
end
N = length(T);

ind1 = true(size(data,1),1);
ind2 = true(N,1);

for j = 1:N
    t = (1:T(j)) + sum(T(1:j-1));
    if isstruct(data)
        c = any(std(data.X(t,:))==0);
    else
        c = any(std(data(t,:))==0);
    end
    ind1(t) = ~c;
    ind2(j) = ~c;
end

if isstruct(data)
    data = data.X(ind1,:);
else
    data = data(ind1,:); 
end
T = T(ind2);

end