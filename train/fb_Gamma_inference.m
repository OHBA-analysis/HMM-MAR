function [Gamma,Xi,L] = fb_Gamma_inference(XX,K,hmm,residuals,slicepoints,constraint)
% inference using normal forward backward propagation

%p = hmm.train.lowrank; do_HMM_pca = (p > 0);
%
% if ~do_HMM_pca && (nargin<4 || isempty(residuals))
%     ndim = size(data.X,2);
%     if ~isfield(hmm.train,'Sind')
%         orders = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
%         hmm.train.Sind = formindexes(orders,hmm.train.S);
%     end
%     if ~hmm.train.zeromean, hmm.train.Sind = [true(1,ndim); hmm.train.Sind]; end
%     residuals =  getresiduals(data.X,T,hmm.train.Sind,hmm.train.maxorder,hmm.train.order,...
%         hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag,hmm.train.zeromean);
% end

if isfield(hmm.train,'distribution') && strcmp(hmm.train.distribution,'logistic')
    order = 0;
else
    order = hmm.train.maxorder;
end
T = size(residuals,1) + order;
Xi = [];

if nargin<5
    slicepoints = [];
end
if nargin<6
    constraint = [];
end

% if isfield(hmm.train,'grouping') && length(unique(hmm.train.grouping))>1
%     i = hmm.train.grouping(n); 
%     P = hmm.P(:,:,i); Pi = hmm.Pi(:,i)'; 
% else 
%     P = hmm.P; Pi = hmm.Pi;
% end
P = hmm.P; Pi = hmm.Pi;
if ~isfield(hmm,'cache'), hmm.cache = []; end

try
    if ~isfield(hmm.train,'distribution') || ~strcmp(hmm.train.distribution,'logistic')
        L = obslike([],hmm,residuals,XX,hmm.cache);
    else
        L = obslikelogistic([],hmm,residuals,XX,slicepoints);
    end
catch
    error('obslike function is giving trouble - Out of precision?')
end

if isfield(hmm.train,'id_mixture') && hmm.train.id_mixture
    Gamma = id_Gamma_inference(L,Pi,order);
    return
end

L(L<realmin) = realmin;

if hmm.train.useMEX 
    [Gamma, Xi, scale] = hidden_state_inference_mx(L, Pi, P, order);
    if any(isnan(Gamma(:))) || any(isnan(Xi(:)))
        clear Gamma Xi scale
        warning('hidden_state_inference_mx file produce NaNs - will use Matlab''s code')
    else
        return
    end
end

scale = zeros(T,1);
alpha = zeros(T,K);
beta = zeros(T,K);

alpha(1+order,:) = Pi.*L(1+order,:);
scale(1+order) = sum(alpha(1+order,:));
alpha(1+order,:) = alpha(1+order,:)/scale(1+order);
for i = 2+order:T
    alpha(i,:) = (alpha(i-1,:)*P).*L(i,:);
    scale(i) = sum(alpha(i,:));		% P(X_i | X_1 ... X_{i-1})
    alpha(i,:) = alpha(i,:)/scale(i);
end

scale(scale<realmin) = realmin;

beta(T,:) = ones(1,K)/scale(T);
for i = T-1:-1:1+order
    beta(i,:) = (beta(i+1,:).*L(i+1,:))*(P')/scale(i);
    beta(i,beta(i,:)>realmax) = realmax;
end

if any(isnan(beta(:)))
    warning(['State time courses became NaN (out of precision). ' ...
        'There are probably extreme events in the data. ' ...
        'Using an approximation..' ])
    Gamma = alpha; out_precision = true; 
else
    Gamma = (alpha.*beta); out_precision = false; 
end

if ~isempty(constraint)
    try
        Gamma(1+order:T,:) = Gamma(1+order:T,:) .* constraint;
    catch
        error(['options.Gamma_constraint must be (trial time X K), ' ...
            ' and all trials must have the same length' ])
    end
end

Gamma = Gamma(1+order:T,:);
Gamma = rdiv(Gamma,sum(Gamma,2));

if out_precision
    Xi = approximateXi(Gamma,size(Gamma,1)+order,hmm);
    Xi = reshape(Xi,[size(Xi,1) size(Xi,2)*size(Xi,3)]);
else
    Xi = zeros(T-1-order,K*K);
    for i = 1+order:T-1
        t = P.*( alpha(i,:)' * (beta(i+1,:).*L(i+1,:)));
        Xi(i-order,:) = t(:)'/sum(t(:));
    end
end
end