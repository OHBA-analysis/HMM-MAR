function hmm = hsupdate(Xi,Gamma,T,hmm)
%
% updates hidden state parameters of an HMM
%
% INPUT:
%
% Xi     probability of past and future state cond. on data
% Gamma  probability of current state cond. on data
% T      length of observation sequences
% hmm    single hmm data structure
%
% OUTPUT
% hmm    single hmm data structure with updated state model probs.
%
% Author: Diego Vidaurre, OHBA, University of Oxford

% if isfield(hmm.train,'grouping')
%     Q = length(unique(hmm.train.grouping));
% else
%     Q = 1;
% end
Q = 1; 
N = length(T); K = hmm.K;
[~,order] = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
embeddedlags = abs(hmm.train.embeddedlags); 
L = order + embeddedlags(1) + embeddedlags(end);
do_clustering = isfield(hmm.train,'cluster') && hmm.train.cluster;

if isempty(Xi) && ~do_clustering % non-exact estimation
    Xi = approximateXi(Gamma,T,hmm);
end
if length(size(Xi))==3, Xi = permute(sum(Xi),[2 3 1]); end

% transition matrix
% if Q>1
%     hmm.Dir2d_alpha = zeros(K,K,Q);
%     hmm.P = zeros(K,K,Q);
%     for i = 1:Q
%         hmm.Dir2d_alpha(:,:,i) = hmm.prior.Dir2d_alpha;
%     end
%     order = hmm.train.maxorder;
%     for n = 1:N
%         i = hmm.train.grouping(n);
%         t = (1:T(n)-1-order) + sum(T(1:n-1)) - (order+1)*(n-1) ;
%         s = permute(sum(Xi(t,:,:)),[2 3 1]);
%         hmm.Dir2d_alpha(:,:,i) = hmm.Dir2d_alpha(:,:,i) + s;
%     end
% else
%     hmm.Dir2d_alpha = permute(sum(Xi),[2 3 1]) + hmm.prior.Dir2d_alpha;
%     hmm.P = zeros(K,K);
% end

if ~do_clustering
    hmm.Dir2d_alpha = Xi + hmm.prior.Dir2d_alpha;
    hmm.P = zeros(K,K);
    for i = 1:Q
        for j = 1:K
            PsiSum = psi(sum(hmm.Dir2d_alpha(j,:,i)));
            for k=1:K
                if ~hmm.train.Pstructure(j,k), continue; end
                hmm.P(j,k,i) = exp(psi(hmm.Dir2d_alpha(j,k,i))-PsiSum);
            end
            hmm.P(j,:,i) = hmm.P(j,:,i) ./ sum(hmm.P(j,:,i));
        end
    end
else
    hmm.Dir2d_alpha = eye(K);
    hmm.P = eye(K);
end

% initial state
if Q==1, hmm.Dir_alpha = hmm.prior.Dir_alpha;
else, hmm.Dir_alpha = repmat(hmm.prior.Dir_alpha',[1 Q]); 
end
i = 1; 
for n = 1:N
    if Q > 1, i = hmm.train.grouping(n); end
    %t = sum(T(1:n-1)) - order*(n-1) + 1;
    if order > 0
        t = sum(T(1:n-1)) - order*(n-1) + 1;
    elseif length(embeddedlags) > 1
        t = sum(T(1:n-1)) - L*(n-1) + 1;
    else
        t = sum(T(1:n-1)) + 1;
    end
    if Q==1 
        hmm.Dir_alpha = hmm.Dir_alpha + Gamma(t,:);
    else
        hmm.Dir_alpha(:,i) = hmm.Dir_alpha(:,i) + Gamma(t,:)';
    end
end

if Q==1
    hmm.Pi = zeros(1,K);
    PsiSum = psi(sum(hmm.Dir_alpha));
    for k = 1:K
        if ~hmm.train.Pistructure(k), continue; end
        hmm.Pi(k) = exp(psi(hmm.Dir_alpha(k))-PsiSum);
    end
    hmm.Pi = hmm.Pi ./ sum(hmm.Pi);
else
    hmm.Pi = zeros(K,Q);
    for i = 1:Q
        PsiSum = psi(sum(hmm.Dir_alpha(:,i)));
        for k = 1:K
            if ~hmm.train.Pistructure(k), continue; end
            hmm.Pi(k,i) = exp(psi(hmm.Dir_alpha(k,i))-PsiSum);
        end
        hmm.Pi(:,i) = hmm.Pi(:,i) ./ sum(hmm.Pi(:,i));
    end
end

end