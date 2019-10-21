function [Y_mult,classlabels] = convertToMultinomial(Y)
% takes data object with mutually exclusive classes and converts to the
% format of multinomial classification. NOTE THAT TEST SET is unchanged

% first check if data_in is a single vector of class labels, in which case
% revert to binary matrix format:
vals = unique(Y(:));
if length(vals)>2 && any(size(Y)==1)
    if ~all(mod(vals,1)==0)
        ME=MException(convertToMultinomial:wrongformat,...
            'Error: labels Y must be either a matrix of binary labels per category, or a vector of integers denoting class membership');
        throw ME;
    end
    data_temp = zeros(length(Y),max(vals));
    if min(vals)<1
        Y=Y+min(vals);
        vals=vals+min(vals);
    end
    for i=1:max(vals)
        data_temp(Y==vals(i),vals(i))=1;
    end
    Y = data_temp;
end

[T_total,ydim] = size(Y);

%setup comparison indices:
[a,b] = find(triu(ones(ydim),1));
% arrange into more interpretable format:
[a,i] = sort(a);b=b(i);
inds=sub2ind([ydim,ydim],a,b);
Y_mult = zeros(T_total,length(inds));
for i=1:T_total
    mat = zeros(ydim);
    activest = find(Y(i,:));
    mat(activest,setdiff(1:ydim,activest))=1;
    mat(setdiff(1:ydim,activest),activest)=-1;
    Y_mult(i,:) = mat(inds)';
end
classlabels=[a,b];
end