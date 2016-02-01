function X = loadfile(file)
if ischar(file)
    if ~isempty(strfind(file,'.mat')), load(file,'X');
    else X = dlmread(file);
    end
else
    X = file;
end
X = X - repmat(mean(X),size(X,1),1);
X = X ./ repmat(std(X),size(X,1),1);
end