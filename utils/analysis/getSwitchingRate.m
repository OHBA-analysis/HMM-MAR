function switchingRate = getSwitchingRate(Gamma,T,options)
% Computes the state switching rate for each session/subject
%
% Gamma can be the probabilistic state time courses (time by states),
%   which can contain the probability of all states or a subset of them,
%   or the Viterbi path (time by 1). 
%
% The parameter 'options' must be the same than the one supplied to the
% hmmmar function for training. 
%
% switchingRate is a vector (sessions/subjects by 1) 
%
% Diego Vidaurre, OHBA, University of Oxford (2017)

if nargin<3, options = struct(); options.Fs = 1; options.downsample = 1; end
if ~isfield(options,'Fs'), options.Fs = 1; end
if ~isfield(options,'downsample'), options.downsample = options.Fs; end

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

if isfield(options,'order') && options.order > 0
    T = ceil(r * T);
    T = T - options.order; 
elseif isfield(options,'embeddedlags') && length(options.embeddedlags) > 1
    d1 = -min(0,options.embeddedlags(1));
    d2 = max(0,options.embeddedlags(end));
    T = T - (d1+d2);
    T = ceil(r * T);
else
    options.order = (sum(T) - size(Gamma,1)) / length(T); 
    T = T - options.order;     
end

if is_vpath % viterbi path
    vpath = Gamma; 
    K = length(unique(vpath));
    Gamma = zeros(length(vpath),K);
    for k = 1:K
       Gamma(vpath==k,k) = 1;   
    end
end

switchingRate = zeros(Nsubj,1);
nt = zeros(Nsubj,1);

for j = 1:N
    t0 = sum(T(1:j-1));
    ind = (1:T(j)) + t0;
    jj = trials2subjects(j);
    if length(ind)==1, continue; end
    switchingRate(jj) = switchingRate(jj) + sum(sum(abs(diff(Gamma(ind,:))),2)) / 2;
    nt(jj) = nt(jj) + length(ind) - 1; 
end
switchingRate = switchingRate ./ nt;

end
