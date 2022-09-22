function [Gamma,Xi,scale] = fb_Gamma_inference(XX,hmm,residuals,slicepoints,constraint,cv)
% inference using normal forward backward propagation
% Gamma: state time courses
% Xi: joint probability for t and t+1
% scale: likelihood

%p = hmm.train.lowrank; do_HMM_pca = (p > 0);
%
% if ~do_HMM_pca && (nargin<4 || isempty(residuals))
%     ndim = size(data.X,2);
%     if ~isfield(hmm.train,'Sind')
%         orders = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
%         hmm.train.Sind = formindexes(orders,hmm.train.S);
%     end
%     if ~hmm.train.zeromean, hmm.train.Sind = [true(1,ndim); hmm.train.Sind]; end
%     residuals =  getresiduals(data.X,T,hmm.train.S,hmm.train.maxorder,hmm.train.order,...
%         hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag,hmm.train.zeromean);
% end

if isfield(hmm.train,'distribution') && strcmp(hmm.train.distribution,'logistic')
    order = 0;
else
    order = hmm.train.maxorder;
end
T = size(residuals,1) + order;
Xi = [];

if nargin<4, slicepoints = []; end
if nargin<5, constraint = []; end
if nargin<6, cv = 0; end

% if isfield(hmm.train,'grouping') && length(unique(hmm.train.grouping))>1
%     i = hmm.train.grouping(n); 
%     P = hmm.P(:,:,i); Pi = hmm.Pi(:,i)'; 
% else 
%     P = hmm.P; Pi = hmm.Pi;
% end
P = hmm.P; Pi = hmm.Pi;
if ~isfield(hmm,'cache'), hmm.cache = []; end

try
    if ~isfield(hmm.train,'distribution') || strcmp(hmm.train.distribution,'Gaussian')
        L = obslike([],hmm,residuals,XX,hmm.cache,cv);
    elseif strcmp(hmm.train.distribution ,'bernoulli')
        L = obslikebernoulli(residuals,hmm);
    elseif strcmp(hmm.train.distribution ,'poisson')
        L = obslikepoisson(residuals,hmm);
    elseif strcmp(hmm.train.distribution ,'logistic')
        L = obslikelogistic([],hmm,residuals,XX,slicepoints);
    end
catch exception
    disp('obslike function is giving trouble - Out of precision?')
    throw(exception)
end

if isfield(hmm.train,'id_mixture') && hmm.train.id_mixture
    [Gamma,scale] = id_Gamma_inference(L,Pi,order);
    return
end

L(L<realmin) = realmin;
L(L>realmax) = realmax;

if hmm.train.useMEX 
    [Gamma, Xi, ~] = hidden_state_inference_mx(L, Pi, P, order);
    if any(isnan(Gamma(:))) || any(isnan(Xi(:)))
        clear Gamma Xi scale
        warning('hidden_state_inference_mx file produce NaNs - will use Matlab''s code')
    else
        return
    end
end

[Gamma,Xi,scale] = fb_Gamma_inference_sub(L,P,Pi,T,order,constraint);

end
