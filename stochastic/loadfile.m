function [X,XX,Y,T] = loadfile(file,T,options,XX_as_list)
% load the file and optionally does (i) embedding and (ii) PCA
if nargin<4, XX_as_list = 0; end
if iscell(file) % T needs to be cell too
    T = cell2mat(T);
    for j=1:length(file)
        if ischar(file{j})
            if ~isempty(strfind(file{j},'.mat')), load(file{j},'X');
            else X = dlmread(file{j});
            end
        else
            X = file{j};
        end
        if j==1
            Xc = zeros(sum(T),size(X,2));
            acc = 0;
        end
        Xc((1:size(X,1)) + acc, :) = X; acc = acc + size(X,1);
    end
    X = Xc; clear Xc 
else
    if ischar(file)
        if ~isempty(strfind(file,'.mat')), load(file,'X');
        else X = dlmread(file);
        end
    else
        X = file;
    end
end
if options.standardise == 1
    for i=1:length(T)
        t = (1:T(i)) + sum(T(1:i-1));
        X(t,:) = X(t,:) - repmat(mean(X(t,:)),length(t),1);
        sdx = std(X(t,:));
        if any(sdx==0)
            error('At least one of the trials/segments/subjects has variance equal to zero');
        end
        X(t,:) = X(t,:) ./ repmat(sdx,length(t),1);
    end
else
    for i = 1:length(T)
        t = (1:T(i)) + sum(T(1:i-1));
        if any(std(X(t,:))==0)
            error('At least one of the trials/segments/subjects has variance equal to zero');
        end
    end
end

% Hilbert envelope
if options.onpower
    X = rawsignal2power(X,T);
end
% Embedding
if length(options.embeddedlags)>1
    [X,T] = embeddata(X,T,options.embeddedlags);
end
% PCA transform
if isfield(options,'A')
    X = X - repmat(mean(X),size(X,1),1); % must center
    X = X * options.A;
    if options.standardise_pc == 1
        X = X - repmat(mean(X),size(X,1),1);
        X = X ./ repmat(std(X),size(X,1),1);
    end
end
if isfield(options,'B'), B = options.B;
else B = []; end
if isfield(options,'V'), V = options.V;
else V = []; end
if nargout>=2
    XX = formautoregr(X,T,options.orders,options.order,options.zeromean,0,B,V);
end
if XX_as_list
    XX_tmp = XX;
    XX = cell(1);
    XX{1} = XX_tmp;
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