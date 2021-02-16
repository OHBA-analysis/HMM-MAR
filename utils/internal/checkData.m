function checkData(X,T,options)
% Check that the data has the right dimensions and no NaNs

if nargin < 3
    L = 0;
elseif isfield(options,'order')
    L = options.order;
elseif isfield(options,'embeddedlags')
    L = 0;
    if any(options.embeddedlags>0), L = L + max(options.embeddedlags); end
    if any(options.embeddedlags<0), L = L + abs(min(options.embeddedlags)); end
else 
    L = 0;
end

if xor(iscell(X),iscell(T))
    error('X and T must be cells, either both or none of them.'); 
end

if iscell(X) 
    if length(T) ~= length(X)
        error('X and T, as cells, must have the same number of elements')
    end
    N = length(T);
    for j = 1:N
        try
            Xj = loadfile_private(X{j}); Tj = T{j};
        catch
            error(['Subject ' num2str(j) ': Error reading the data'])
        end
        if size(Xj,1) ~= sum(Tj)
            error(['Subject ' num2str(j) ': The size of the data ' ...
                'and the specified T does not coincide'])
        end
        if any(Tj <= L)
            error(['Subject ' num2str(j) ': The size of the data ' ...
                'is too short for at least one of the trials'])
        end
        if any(isnan(Xj(:))) || any(isinf(Xj(:)))
            error(['Subject ' num2str(j) ': NaN or Inf were found'])
        end
    end
else
    if isstruct(X), X = X.X; end
    if size(X,1) ~= sum(T)
        error(['The size of the data ' ...
            'and the specified T does not coincide'])
    end
    if any(T <= L)
        error(['The size of the data ' ...
            'is too short for at least one of the trials or subjects'])
    end
    if any(isnan(X(:))) || any(isinf(X(:)))
        error('NaN or Inf were found')
    end
end

end


function X = loadfile_private(file)

if ischar(file)
    fsub = file;
    loadfile_sub;
elseif iscell(file)
    error('There are cells within cells in the data')
else
    X = file;
end

end
