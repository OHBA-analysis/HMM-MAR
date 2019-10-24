function [preds_K,hard_class_pred] = multinomLogRegPred(preds_N)
% function to convert a mmultinomial logistic regression data back to class
% probabilities. input should be a [T x nTr x N] matrix of scores, where
% N=(K.^2-K)/2 possible permutations

[T,nTr,N] = size(preds_N);
K = 0.5*(1+sqrt(1+8*N)); %solving by quadratic polynomial - this must give round number
if mod(K,1)~=0
    ME = MException(multinomialLogReg:dimensionproblem,'Error: dimensions of multinomial predictions not compatible with whole number of classes');
    throw ME;
end

%setup comparison indices:
[a,b] = find(triu(ones(K),1));
% arrange into more interpretable format:
[a,i] = sort(a);b=b(i);
inds=sub2ind([K,K],a,b);
inds2=sub2ind([K,K],b,a);
output_scores = zeros(T,nTr,K,K);
output_scores(:,:,inds) = log_sigmoid(preds_N);
output_scores(:,:,inds2) = log_sigmoid(-preds_N);
%output_scores = log_sigmoid(output_scores);
output_scores(:,:,find(eye(K))) = 1;

classLikelihood = squeeze(prod(output_scores,4));
preds_K = classLikelihood ./ repmat(sum(classLikelihood,3),1,1,K);
% we store preds in a matrix where (i,j)th entry is probability for i vs j
% for t=1:T
%     for iTr=1:nTr
%         
%         
%         output_scores = output_scores - output_scores';
%         output_scores = log_sigmoid(output_scores);
%         preds_K(t,iTr
%     end
% end

%also return a binary class assignment:
[~,hard_class_pred] = max(preds_K,[],3);
end