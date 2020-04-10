function [X,XX,Y,T] = loadfile(file,T,options)
% load the file and does some optional preprocessing

if iscell(file) % T needs to be cell too
    T = cell2mat(T);
    for j = 1:length(file)
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

% Standardise data and control for ackward trials
if isfield(options,'standardise')
    X = standardisedata(X,T,options.standardise);
end
% Filtering
if isfield(options,'filter') && ~isempty(options.filter)
    X = filterdata(X,T,options.Fs,options.filter);
end
% Detrend data
if isfield(options,'detrend') && options.detrend
    X = detrenddata(X,T);
end
% Leakage correction
if isfield(options,'leakagecorr') && options.leakagecorr ~= 0
    X = leakcorr(X,T,options.leakagecorr);
end
% Hilbert envelope
if isfield(options,'onpower') && options.onpower
    X = rawsignal2power(X,T);
end
% Leading Phase Eigenvectors
if isfield(options,'leida') && options.leida
    X = leadingPhEigenvector(X,T);
end
% PCA transform (before embedded, i.e. drawing only from space corr)
if isfield(options,'As') && ~isempty(options.As)
    X = bsxfun(@minus,X,mean(X)); % must center
    X = X * options.As;
end
% Embedding
if isfield(options,'embeddedlags') && length(options.embeddedlags)>1
    [X,T] = embeddata(X,T,options.embeddedlags);
end
% PCA transform
if isfield(options,'A') && ~isempty(options.A)
    X = bsxfun(@minus,X,mean(X)); % must center
    X = X * options.A;
    % Standardise principal components and control for ackward trials
    X = standardisedata(X,T,options.standardise_pc);
end
% Downsampling
if isfield(options,'downsample') && options.downsample > 0
    [X,T] = downsampledata(X,T,options.downsample,options.Fs);
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