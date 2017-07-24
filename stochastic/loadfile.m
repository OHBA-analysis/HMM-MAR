function [X,XX,Y,T] = loadfile(file,T,options,XX_as_list)
% load the file and does some optional preprocessing
if nargin<4, XX_as_list = 0; end
if ~isfield(options,'downsample'), options.downsample = 0; end
if iscell(file) % T needs to be cell too
    T = cell2mat(T);
    for j=1:length(file)
        if ischar(file{j})
            fsub = file{j};
            loadfile_sub;
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
        fsub = file;
        loadfile_sub;
    else
        X = file;
    end
end

% Filtering
if ~isempty(options.filter)
    X = filterdata(X,T,options.Fs,options.filter);
end
% Detrend data
if options.detrend
    X = detrenddata(X,T);
end
% Standardise data and control for ackward trials
X = standardisedata(X,T,options.standardise);
% Hilbert envelope
if options.onpower
    X = rawsignal2power(X,T);
end
% PCA transform (before embedded, i.e. drawing only from space corr)
if isfield(options,'As') && ~isempty(options.As)
    X = bsxfun(@minus,X,mean(X)); % must center
    X = X * options.As;
end
% Embedding
if length(options.embeddedlags)>1
    [X,T] = embeddata(X,T,options.embeddedlags);
end
% PCA transform
if isfield(options,'A')
    X = bsxfun(@minus,X,mean(X)); % must center
    X = X * options.A;
    % Standardise principal components and control for ackward trials
    X = standardisedata(X,T,options.standardise_pc);
end
% Downsampling
if options.downsample > 0
    [X,T] = downsampledata(X,T,options.downsample,options.Fs);
end
% Data transformation for crosstermsonly==1
if isfield(options,'crosstermsonly') && options.crosstermsonly
    ndim = size(X,2); 
    Ttmp = T;
    T = T + 1;
    Xtmp = zeros(sum(T),2*ndim);
    for n=1:length(T)
        t1 = (1:T(n)) + sum(T(1:n-1));
        t2 = (1:Ttmp(n)) + sum(Ttmp(1:n-1));
        Xtmp(t1(2:end),1:ndim) = X(t2,:);
        Xtmp(t1(1:end-1),(ndim+1):end) = X(t2,:);
    end
    X = Xtmp; 
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