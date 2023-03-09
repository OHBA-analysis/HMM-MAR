function folds = cvfolds_FK(Y,family,CVscheme,allcs)
% allcs can be a N x 1 vector with family memberships; an (N x N) matrix
% with family relationships; or empty.
% If family is 'multinomial', it will stratify the estimation
% Diego Vidaurre
%
% adapted to randomise folds taking family structure into account
% NOTE: when doing this, folds can end up having different sizes
% Christine Ahrends

rng('shuffle')
if nargin<4, allcs = []; end
is_cs_matrix = (size(allcs,2) == 2);

[N,q] = size(Y);

if CVscheme==0, nfolds = N;
else nfolds = CVscheme;
end

if isempty(allcs)
    folds = cell(nfolds,1);
    if nfolds==N
        for j = 1:N, folds{j} = j; end
        return
    elseif strcmpi(family,'multinomial')
        if q > 1, Y = nets_class_mattovec(Y); end
        rng('shuffle')
        c = cvpartition(Y,'KFold',nfolds);
    else
        rng('shuffle')
        c = cvpartition(N,'KFold',nfolds);
    end
    for k = 1:nfolds
        folds{k} = find(c.test(k));
    end
    return
end

if ~strcmpi(family,'multinomial')
    if length(unique(Y)) <= 3
        Y = nets_class_vectomat(Y);
        q = size(Y,2);
    end
elseif strcmpi(family,'multinomial') && q==1
    Y = nets_class_vectomat(Y); q = size(Y,2);
end

do_stratified = (strcmpi(family,'multinomial') || size(Y,2)>1);
folds = cell(nfolds,1); grotDONE = false(N,1);
counts = zeros(nfolds,q); Scounts = sum(Y);
foldsDONE = false(nfolds,1); foldsUSED = false(nfolds,1);
j = 1;

fold_size = round(N/nfolds,0);
for iii = 1:fold_size+1
    randfold_ind(iii,:) = randperm(nfolds); % create random array of folds so that each fold appears exactly the same number of times
end
randfold_ind = randfold_ind';
randfold_vec = randfold_ind(:); % use random array of folds as vector to be indexed by j
while j<=N
    if grotDONE(j), j = j+1; continue; end
    Jj = j;
    % pick up all of this family
    if is_cs_matrix
        if size(find(allcs(:,1)==j),1)>0, Jj=[j allcs(allcs(:,1)==j,2)']; end
    else
        if allcs(j)>0
            Jj = find(allcs==allcs(j))';
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
    else 
        if j < round(N*0.8,0) % for the first 80% of examples, randomly attribute the families into any fold
            %ii = randi(nfolds);
            ii = randfold_vec(j);                
        else % choose the fold with less examples (only in the last 10% of examples)
            [~,ii] = min(counts);
            counts(ii) = counts(ii) + length(Jj);
        end
    end
    % update folds, and the other indicators
    folds{ii} = [ folds{ii} Jj ];
    grotDONE(Jj) = true;
    if length(folds{ii}) >= N/nfolds, foldsDONE(ii) = true; end
    foldsUSED(ii) = true;
    j = j+1;
end

folds = folds(foldsUSED);

end
