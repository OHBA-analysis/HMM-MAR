function [FO,ntrials] = getFractionalOccupancy (Gamma,T,options,dim,alignment)
% 
% Computes the fractional occupancy  
% - across trials if dim==1 (i.e. FO is time by no.states)
%   This is indicating how the fractional occupancy evolves as a function
%   of time across trials.
% - across time if dim==2 (i.e. FO is no.trials/subjects by no.states) 
%   This is indicating how much of each state each subject has.
% (default, dim=2)
% 
% The parameter 'options' must be the same than the one supplied to the
% hmmmar function for training. The structure hmm.train can also be
% supplied
% 
% If dim==1, how is the across-time average defined when trials have not
% the same length? The trials can be aligned in two different ways: such
% that they all start at the same time point, or such that they all finish
% at the same time point. The parameter 'alignment' defines this, and can
% take values 'start' or 'finish'.
%
% If dim==1, also, ntrials will return a (time by 1) vector telling how many
% trials were used to compute the average at that time point. This is
% useful when trials have different lengths. 
%
% Note: this can be applied to the data as well, if you want to look at the
% evoked response. 
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2016)

if nargin<3, options = struct(); options.Fs = 1; options.downsample = 1; end
if ~isfield(options,'Fs'), options.Fs = 1; end
if ~isfield(options,'downsample'), options.downsample = options.Fs; end
if nargin<4, dim = 2; end
if nargin<5 && dim == 1, alignment = 'start'; end

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
elseif dim == 2
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

if isfield(options,'tuda') && options.tuda
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
    K = length(unique(vpath));
    Gamma = zeros(length(vpath),K);
    for k = 1:K
       Gamma(vpath==k,k) = 1;   
    end
else
    K = size(Gamma,2); 
    %Gamma = Gamma > (2/3);
end

if dim == 2
    FO = zeros(Nsubj,K);
    for j = 1:N
        t0 = sum(T(1:j-1));
        ind = (1:T(j)) + t0;
        jj = trials2subjects(j);
        if length(ind) > 1
            FO(jj,:) = FO(jj,:) + sum(Gamma(ind,:));
        else
            FO(jj,:) = FO(jj,:) + Gamma(ind,:);
        end
    end
    FO = FO ./ repmat(sum(FO,2),1,K);
    ntrials = [];
else
    if all(T==T(1)) 
        Gamma = reshape(Gamma,[T(1),N,K]);
        FO = squeeze(mean(Gamma,2));
        ntrials = N * ones(T(1),1);
    else
        maxT = max(T);
        FO = zeros(maxT,K);
        ntrials = zeros(maxT,1);
        for j = 1:N
            t0 = sum(T(1:j-1));
            ind_gamma = (1:T(j)) + t0;
            if strcmpi(alignment,'start')
                ind_FO = 1:length(ind_gamma);
            else
                ind_FO = (1:length(ind_gamma)) + (maxT-length(ind_gamma));
            end
            FO(ind_FO,:) = FO(ind_FO,:) + Gamma(ind_gamma,:);
            ntrials(ind_FO) = ntrials(ind_FO) + 1;
        end
        FO = FO ./ repmat(ntrials,1,K);
    end
end

end

