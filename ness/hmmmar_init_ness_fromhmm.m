function Gamma = hmmmar_init_ness_fromhmm(Gamma,T,options)
%
% Initialise the NESS chain using a previously run HMM, with K+1 states
%
% INPUT
% Gamma  Initialised state time courses
% T          length of observation sequence
% options    structure with the training options  
%
% OUTPUT
% Gamma     p(state given X)
%
% Author: Diego Vidaurre, Aarhus University / Oxford , 2021

if ~isfield(options,'maxorder')
    [~,order] = formorders(options.order,options.orderoffset,...
        options.timelag,options.exptimelag);
    options.maxorder = order;
end
L = options.maxorder; K = options.K; 
if isfield(options,'ness_initfromhmm_crit'), crit = options.ness_initfromhmm_crit;
else, crit = 1; 
end

if size(Gamma,2) ~= (K+1)
    error('The initialised state time courses must have K+1 states')
end
if size(Gamma,1) ~= sum(sum(T)-L*length(T))
    error('The number of time points in the state time courses is incorrect')
end

if crit==1 % the one with maximum occupancy is the default
    [~,order] = sort(sum(Gamma));
    Gamma = Gamma(:,order);
    Gamma = Gamma(:,1:end-1);
else % the one with the minimum temporal variance is the default (not v good)
   [~,k] = min(var(Gamma));
   Gamma = Gamma(:,setdiff(1:(K+1),k));
end

end
