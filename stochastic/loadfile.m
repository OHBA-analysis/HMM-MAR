function [X,XX,Y,T] = loadfile(file,T,options)
% load the file and optionally does (i) embedding and (ii) PCA 
if ischar(file)
    if ~isempty(strfind(file,'.mat')), load(file,'X');
    else X = dlmread(file);
    end
else
    X = file;
end
if options.standardise == 1
    for i=1:length(T)
        t = (1:T(i)) + sum(T(1:i-1));
        X(t,:) = X(t,:) - repmat(mean(X(t,:)),length(t),1);
        X(t,:) = X(t,:) ./ repmat(std(X(t,:)),length(t),1);
    end
end
if length(options.embeddedlags)>1
    [X,T] = embeddata(X,T,options.embeddedlags);
end
if options.pca > 0 && isfield(options,'A')
    X = X - repmat(mean(X),size(X,1),1); % must center
    X = X * options.A;
end
if isfield(options,'B'), B = options.B;
else B = []; end
if isfield(options,'V'), V = options.V;
else V = []; end
if nargout>=2
    XX = formautoregr(X,T,options.orders,options.order,options.zeromean,0,B,V);
end
if nargout>=3
    Y = zeros(sum(T)-length(T)*options.order,size(X,2));
    for in=1:length(T)
        t0 = sum(T(1:in-1));
        t = sum(T(1:in-1)) - (in-1)*options.order;
        Y(t+1:t+T(in)-options.order,:) = X(t0+options.order+1:t0+T(in),:);
    end
end
end