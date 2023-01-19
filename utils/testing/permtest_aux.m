function pval = permtest_aux(X,D,Nperm,confounds,pairs,index_subjects)
% permutation testing routine (through regression)
% X: data
% D: design matrix
% Nperm: no. of permutations
% confounds: to be regressed out
% 
%
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
paired = (nargin>4) && ~isempty(pairs);
permute_per_subject = (nargin>5) && ~isempty(index_subjects);

%D = bsxfun(@minus,D,mean(D));   
X = bsxfun(@minus,X,mean(X));  
grotperms = zeros(Nperm,p);
proj = (D' * D + 0.001 * eye(size(D,2))) \ D';  

for perm = 1:Nperm
    if perm==1
        Xin = X;
    elseif permute_per_subject
        Xin = zeros(size(X));
        Nsubj = max(index_subjects);
        for j = 1:Nsubj
           jj = find(index_subjects == j); 
           Ntrials = length(jj);
           r = randperm(Ntrials);
           Xin(jj,:) = X(jj(r),:);
        end
    elseif paired
        r = 1:N;
        for j = 1:max(pairs)
            jj = find(pairs==j);
            if length(jj) < 2, continue; end
            if rand<0.5
                r(jj) = r(jj([2 1]));
            end
        end
        Xin = X(r,:);
    else
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
