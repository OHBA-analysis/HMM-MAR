function valid_dims = computeValidDimensions(data,S)
% Evaluate a matrix of connections S and return which elements are to be
% analysed for variance and standardisation.
if all(S(:))==1
    valid_dims=[1:length(S)];
else
    %implies a TUDA/TUCA style setup:
    lastdatadim = find(diff(any(options.S==1)),1);
    % check if intercept term present:
    if isstruct(data)
         X = data.X;
    else
         X = data;
    end
    if var(X(:,lastdatadim))<1e-4
        lastdatadim=lastdatadim-1;
    end
    valid_dims=[1:lasdatadim];
    
end


end