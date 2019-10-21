function S = getStartFinish_StateVisits (vp,T,options)
% Decomposes a Viterbi path into a list of visits, each characterised by  
% start time point and finish time point.
%
% S is a (N x K) cell, where N is the number of data time series 
% (length of T) and K is the number of states; each element of S contains 
% a matrix, which n-th row corresponds to the n-th visit to the
% corresponding state, with two columns (start and finish time points). 
% The values of these matrices are in time points (not seconds), and are
% relative to the length of the state time courses, which will be a bit
% shorter if a MAR or a time-delay embedded observation model were used.
%
% T must be a vector with the length of each time series, and vp is the
% Viterbi path as returned by hmmmar or hmmdecode. 
% The parameter 'options' must be the same than the one supplied to the
% hmmmar function for training. 
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2018)
 
if nargin<3, options = struct(); options.Fs = 1; options.downsample = 1; end
if ~isfield(options,'Fs'), options.Fs = 1; end
if ~isfield(options,'downsample'), options.downsample = options.Fs; end

if nargin < 2, T = length(vp); end 

if iscell(T)
    if size(T,1) == 1, T = T'; end
    for i = 1:length(T)
        if size(T{i},1)==1, T{i} = T{i}'; end
    end
    T = cell2mat(T);
end
N = length(T);

r = 1; 
if isfield(options,'downsample') && options.downsample>0
    r = (options.downsample/options.Fs);
end

if isfield(options,'order') && options.order > 0
    T = ceil(r * T);
    T = T - options.order; 
elseif isfield(options,'embeddedlags') && length(options.embeddedlags) > 1
    d1 = -min(0,options.embeddedlags(1));
    d2 = max(0,options.embeddedlags(end));
    T = T - (d1+d2);
    T = ceil(r * T);
end

K = length(unique(vp));
S = cell(N,K);

for j = 1:N
    t0 = sum(T(1:j-1));
    ind = (1:T(j)) + t0;
    vp_j = vp(ind); 
    for k = 1:K
       vp_jk = zeros(T(j),1);
       vp_jk(vp_j==k) = 1; 
       vp_jk_diff = diff(vp_jk);
       Nvisits = sum(vp_jk_diff==1) + (vp_jk(1)==1); 
       S{j,k} = zeros(Nvisits,2);
       if vp_jk(1)==1 % first visit is to this state
           S{j,k}(1,1) = 1; 
           S{j,k}(2:end,1) = find(vp_jk_diff==1)+1;
       else
           S{j,k}(:,1) = find(vp_jk_diff==1)+1;
       end
       if vp_jk(end)==1 % last visit is to this state
           S{j,k}(1:end-1,2) = find(vp_jk_diff==(-1));
           S{j,k}(end,2) = T(j);
       else
           S{j,k}(:,2) = find(vp_jk_diff==(-1));
       end
    end       
end

end
