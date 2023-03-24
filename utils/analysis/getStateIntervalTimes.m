function intervals = getStateIntervalTimes (Gamma,T,options,threshold,threshold_Gamma,do_concat)
% Computes the interval times for the state time courses  
% 
% Gamma can be the probabilistic state time courses (time by states),
%   which can contain the probability of all states or a subset of them,
%   or the Viterbi path (time by 1). 
%
% In the first case, threshold_Gamma is used to define when a state is active. 
%
% The parameter threshold is used to discard intervals that are too
%   short (deemed to be spurious); these are discarded when they are shorter than 
%   'threshold' time points
%
% The parameter 'options' must be the same than the one supplied to the
% hmmmar function for training. 
%
% Intervals is then a cell (subjects by states), where each element is a
%   vector of interval times
%
% if do_concat is specified, the interval times are concatenated across
% segments of data, so that interval times is a cell (1 by states)
% 
% Diego Vidaurre, OHBA, University of Oxford (2016)

if nargin<3, options = struct(); options.Fs = 1; options.downsample = 1; end
if ~isfield(options,'Fs'), options.Fs = 1; end
if ~isfield(options,'downsample'), options.downsample = options.Fs; end
if nargin<4 || isempty(threshold), threshold = 0; end
if nargin<5 || isempty(threshold_Gamma), threshold_Gamma = (2/3); end
if nargin<6 || isempty(do_concat), do_concat = true; end

is_vpath = (size(Gamma,2)==1 && all(rem(Gamma,1)==0)); 
if iscell(T)
    if size(T,1) == 1, T = T'; end
    for i = 1:length(T)
        if size(T{i},1)==1, T{i} = T{i}'; end
    end
    Nsubj = length(T);
    trials2subjects = zeros(length(cell2mat(T)),1); ii = 1; 
    for i = 1:length(T)
        Ntrials = length(T{i});
        trials2subjects(ii:ii+Ntrials-1) = i;
        ii = ii + Ntrials;
    end
    T = cell2mat(T);
else 
    Nsubj = length(T);
    trials2subjects = 1:Nsubj;
end
N = length(T);

r = 1; 
if isfield(options,'downsample') && options.downsample>0
    r = (options.downsample/options.Fs);
end

if ~isfield(options,'order') && ~isfield(options,'embeddedlags')
   options.order = (sum(T) - size(Gamma,1)) / length(T);
end

if isfield(options,'tuda') && options.tuda | ...
        isfield(options, 'order') && options.order == 0
    T = ceil(r * T);
elseif isfield(options,'order') && options.order > 0
    T = ceil(r * T);
    T = T - options.order; 
elseif isfield(options,'embeddedlags') && length(options.embeddedlags) > 1
    d1 = -min(0,options.embeddedlags(1));
    d2 = max(0,options.embeddedlags(end));
    T = T - (d1+d2);
    T = ceil(r * T);
end

if is_vpath % viterbi path
    vpath = Gamma; 
    if options.dropstates==1
        K = length(unique(vpath));
    else
        K = options.K;
    end
    Gamma = zeros(length(vpath),K);
    for k = 1:K
       Gamma(vpath==k,k) = 1;   
    end
else
    warning(['Using the Viterbi path is here recommended instead of the state ' ...
        'probabilistic time courses (Gamma)'])
    K = size(Gamma,2); 
    Gamma = Gamma > threshold_Gamma;
end

if ~do_concat
    intervals = cell(Nsubj,K);
    for j = 1:N
        t0 = sum(T(1:j-1));
        ind = (1:T(j)) + t0;
        if length(ind)==1, continue; end
        jj = trials2subjects(j);
        for k=1:K
            t = find(Gamma(ind,k)==1,1);
            if isempty(t), continue; end
            intervals{jj,k} = [intervals{jj,k} aux_k(Gamma(ind(t:end),k),threshold)];
        end
    end
else
    intervals = cell(1,K);
    for j = 1:N
        t0 = sum(T(1:j-1));
        ind = (1:T(j)) + t0;
        if length(ind)==1, continue; end
        for k=1:K
            t = find(Gamma(ind,k)==1,1);
            if isempty(t), continue; end
            intervals{k} = [intervals{k} aux_k(Gamma(ind(t:end),k),threshold)];
        end
    end
end

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
