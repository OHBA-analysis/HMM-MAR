function [X,XX,Y] = loadfile(file,T,options)
if ischar(file)
    if ~isempty(strfind(file,'.mat')), load(file,'X');
    else X = dlmread(file);
    end
else
    X = file;
end
X = X - repmat(mean(X),size(X,1),1);
X = X ./ repmat(std(X),size(X,1),1);
if nargout>=2
    XX = formautoregr(X,T,options.orders,options.order,options.zeromean);
end
if nargout==3
    Y = zeros(sum(T)-length(T)*options.order,size(X,2)); 
    for in=1:length(T)
        t0 = sum(T(1:in-1));
        t = sum(T(1:in-1)) - length(T)*options.order;
        Y(t+1:t+T(in)-options.order,:) = X(t0+options.order+1:t0+T(in),:);
    end
end
end