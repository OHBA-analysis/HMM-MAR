function [WishTrace,X_coded_vals] = computeWishTrace(hmm,regressed,XX,C,k)
X = XX(:,~regressed);
if any(hmm.train.S(:)~=1) && length(unique(X))<5
    % regressors are low dim categorical - compute and store in cache for
    % each regressor type - convert to binary code:
    X_coded = X*[2.^(1:size(X,2))]';
    X_coded_vals = unique(X_coded);
    validentries = logical(hmm.train.S(:)==1);
    WishTrace = zeros(1,length(X_coded_vals));
    B_S = hmm.state(k).W.S_W(validentries,validentries);
    for i=1:length(X_coded_vals)
        t_samp = find(X_coded==X_coded_vals(i),1);
        WishTrace(i) = trace(kron(C(regressed,regressed),X(t_samp,:)'*X(t_samp,:))*B_S);
    end
else
    WishTrace =[];
    X_coded_vals=[];
end
end