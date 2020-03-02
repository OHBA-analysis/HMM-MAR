function valid_dims = computeValidDimensions(data,options)
% Evaluate a matrix of connections S and return which elements are to be
% analysed for variance and standardisation.
S = options.S;
if ~isfield(options,'distribution') || strcmp(options.distribution,'Gaussian') 
    if all(S(:)==1)
        valid_dims = [1:length(S)]; % note this omits first dimension to allow TUDA style use of intercepts
    else
        %implies a TUDA/TUCA style setup:
        lastdatadim = find(diff(any(S==1)),1);
        % check if intercept term present:
        if isstruct(data)
             X = data.X;
        else
             X = data;
        end
        if var(X(:,lastdatadim))==0
            lastdatadim = lastdatadim-1;
        end
        valid_dims = [1:lastdatadim];
    end
else
    valid_dims = [];
end

end