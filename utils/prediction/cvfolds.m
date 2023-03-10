function foldsi = cvfolds(Y,CVscheme,allcs,perm)
% allcs can be a N x 1 vector with family memberships, an (N x N) matrix
% with family relationships, or empty.
% perm: whether to permute subjects when assigning them to folds or not (if
% not, they will be assigned (accounting for family structure) in the order
% they appear
% Diego Vidaurre
% edits Christine Ahrends

if nargin<3, allcs = []; end
is_cs_matrix = (size(allcs,2) == 2);

if nargin<4
    perm=1;
end

[N,q] = size(Y); 

if perm==1
    indperm = randperm(N);
    Y = Y(indperm,:);
end

if CVscheme==0, nfolds = N;
else nfolds = CVscheme;
end

if isempty(allcs)
    folds = cell(nfolds,1);
    if nfolds==N
        for j = 1:N, folds{j} = j; end
        return
    else
        if q > 1
            Y = nets_class_mattovec(Y);
            c = cvpartition(Y,'KFold',nfolds);
        elseif length(unique(Y)) <= 4
            c = cvpartition(Y,'KFold',nfolds);
        else
            c = cvpartition(N,'KFold',nfolds);
        end
    end
    for k = 1:nfolds
        folds{k} = find(c.test(k));
    end
    return
end

if q == 1 && length(unique(Y)) <= 4
    Y = nets_class_vectomat(Y);
    q = size(Y,2);
end

do_stratified = q > 1;
folds = cell(nfolds,1); grotDONE = false(N,1);
counts = zeros(nfolds,q); Scounts = sum(Y);
foldsDONE = false(nfolds,1); foldsUSED = false(nfolds,1);
j = 1;

while j<=N
    if grotDONE(j), j = j+1; continue; end
    Jj = j;
    % pick up all of this family
    if is_cs_matrix
        if perm==1
            for iii = 1:size(allcs,1)
                for jjj = 1:N
                    if allcs(iii,1)==jjj
                        allcs1(iii,1)=indperm(jjj);
                    end
                    if allcs(iii,2)==jjj
                        allcs1(iii,2)=indperm(jjj);
                    end
                end
            end
        else
            allcs1 = allcs;
        end
        if size(find(allcs1(:,1)==j),1)>0, Jj=[j allcs1(allcs1(:,1)==j,2)']; end
    else
        if perm==1
            allcs1 = allcs(indperm);
        else
            allcs1 = allcs;
        end
        if allcs1(j)>0
            Jj = find(allcs1==allcs1(j))';
        end
    end; Jj = unique(Jj);
    if do_stratified
        % how many of each class there is
        if length(Jj)>1, countsI = sum(Y(Jj,:));
        else, countsI = Y(Jj,:);
        end
        % which fold is furthest from the wished class counts?
        d = -Inf(nfolds,1);
        for i = 1:nfolds
            if foldsDONE(i), continue; end
            c = counts(i,:) + countsI; 
            d(i) = sum( ( Scounts - c ) );
        end
        % to break the ties, choose the fold with less examples
        m = max(d); ii = (d==m);
        counts2 = sum(counts,2); counts2(~ii) = Inf; 
        [~,ii] = min(counts2);
        counts(ii,:) = counts(ii,:) + countsI;
    else % just choose the fold with less examples
        [~,ii] = min(counts);
        counts(ii) = counts(ii) + length(Jj);
    end
    % update folds, and the other indicators
    folds{ii} = [ folds{ii} Jj ];
    grotDONE(Jj) = true;
    if length(folds{ii}) >= N/nfolds, foldsDONE(ii) = true; end
    foldsUSED(ii) = true;
    j = j+1;
end

folds = folds(foldsUSED);
if perm==1
    foldsi = cell(size(folds));
    for ii = 1:size(folds,1)
        for jjj = 1:numel(folds{ii})
            for n = 1:N
                if folds{ii}(jjj) == n
                    foldsi{ii}(jjj) = find(indperm==n);
                end
            end
        end
    end
else
    foldsi = folds;
end
                


end


function Ym = nets_class_vectomat(Y,classes)
N = length(Y);
if nargin<2, classes = unique(Y); end
q = length(classes);
Ym = zeros(N,q);
for j=1:q, Ym(Y==classes(j),j) = 1; end
end


function Y = nets_class_mattovec(Ym,classes)
if nargin<2, q = size(Ym,2); classes = 1:q; 
else q = length(classes); 
end
Y = zeros(size(Ym,1),1);
for j=1:q, Y(Ym(:,j)>.5) = classes(j); end
end
