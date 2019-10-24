function onsets = getStateOnsets(vpath,T,Hz,K)
% Finds the onset times of the states for each session.
%
% vpath must be the Viterbi path, and T is the length of the sessions
% Hz is the sampling frequency
% onsets is a cell (subjects by states), where each element is a
%   vector of onset times
% K (optional) prespecifies the number of states. This is useful when vpath
% does not visit the entire repertoire of states
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2017)


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
T = T - (sum(T)-length(vpath))/length(T);

N = length(T);
%val = unique(vpath)';
if nargin<4 || isempty(K)
    K = max(val);
end
val = 1:K;
    
onsets = cell(N,K);
for n=1:N
    t = sum(T(1:n-1)) + (1:T(n));
    for ik = 1:K
        k = val(ik);
        vpathk = zeros(length(t),1);
        vpathk(vpath(t)==k) = 1;
        dvpathk = diff(vpathk);
        dvpathk(dvpathk==-1) = 0;
        dvpathk = [vpathk(1)==1; dvpathk];
        onsets{n,ik} = find(dvpathk==1) / Hz;
    end
end

end