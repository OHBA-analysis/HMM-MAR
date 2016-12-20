function I = randpermNK(N,K) 
% N is the number of trials/subjects
% K is the number of groups
if K==1, 
    I = cell(1);
    I{1} = randperm(N); 
    return; 
end
np=(N-mod(N,K))/K; % number of elements per group
[~,idx]=sort(rand(N,1));
i=1;
j=1;
I={};
while 1
    I{j}=idx(i:i+np-1,1);
    if N-(i+np)+1 < np
        I{j} = [I{j}; idx(i+np:end,1)];
        break
    end
    i=i+np;
    j=j+1;
end
end 