function pval = permtest_aux(X,D,Nperm,confounds)
% permutation testing routine (through regression)
% X: data
% D: design matrix
% Nperm: no. of permutations
% confounds
% grouping: the first grouping(1) rows belong to one group, 
%    the next grouping(2) rows belong to the second group, 
%    and so on and so on - DEPRECATED
% Diego Vidaurre

[N,p] = size(X);
%if nargin<4, grouping = []; end
if (nargin>3) && ~isempty(confounds)
    confounds = confounds - repmat(mean(confounds),N,1);
    X = bsxfun(@minus,X,mean(X));   
    X = X - confounds * pinv(confounds) * X;
    %D = bsxfun(@minus,D,mean(D));   
    %D = D - confounds * pinv(confounds) * D;    
end

%D = bsxfun(@minus,D,mean(D));   
X = bsxfun(@minus,X,mean(X));  
grotperms = zeros(Nperm,p);
proj = (D' * D + 0.001 * eye(size(D,2))) \ D';  

for perm=1:Nperm
    if perm==1
        Xin = X;
    else
        %if ~isempty(grouping)
        %    Xin = zeros(size(X));
        %    for gr = 1:length(grouping)
        %        jj = (1:grouping(gr)) + sum(grouping(1:gr-1));
        %        r = randperm(grouping(gr));
        %        Xin(jj,:) = X(jj(r),:);
        %    end
        %else
        %    r = randperm(N);
        %    Xin = X(r,:);
        %end
        Xin = X(randperm(N),:);
    end
    beta = proj * Xin;
    grotperms(perm,:) = sqrt(sum((D * beta - Xin).^2)); 
end

pval = zeros(p,1);
for j = 1:p
    pval(j) = sum(grotperms(:,j)<=grotperms(1,j)) / (Nperm+1);
end
    
end
