function onsets = getStateOnsets(vpath,T,Hz)

if size(vpath,2)>1 
   error('Viterbi path is needed') 
end
if nargin<3, Hz = 1; end

if iscell(T)
    if size(T,1)==1, T = T'; end
    for i = 1:length(T)
        if size(T{i},1)==1, T{i} = T{i}'; end
    end
    T = cell2mat(T);
end

N = length(T); 
val = unique(vpath)';
K = length(val);

onsets = cell(N,K); 
for n=1:N
    t = sum(T(1:n-1)) + (1:T(n));
    for k = val
        vpathk = zeros(length(t),1);
        vpathk(vpath(t)==k) = 1;
        dvpathk = diff(vpathk);
        dvpathk(dvpathk==-1) = 0;
        dvpathk = [vpathk(1)==1; dvpathk];
        onsets{n,k} = find(dvpathk==1) / Hz;   
    end
end

end