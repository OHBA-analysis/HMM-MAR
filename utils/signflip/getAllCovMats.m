function covmats_unflipped = getAllCovMats(data,T,options)

if ~isfield(options,'maxlag'), options.maxlag = 10; end
if ~isfield(options,'partial'), options.partial = 0; end
if ~isfield(options,'standardise'), options.standardise = 1; end

N = length(T);
if iscell(data)
    for j = 1:N
        if ischar(data{j})
            fsub = data{j};
            loadfile_sub;
        else
            X = data{j};
        end
        Tj = T{j}; Nj = length(Tj);
        if options.standardise
            for jj = 1:Nj
                ind = (1:Tj(jj)) + sum(Tj(1:jj-1));
                X(ind,:) = bsxfun(@minus,X(ind,:),mean(X(ind,:)));
                sd = std(X(ind,:));
                if any(sd==0)
                    error('At least one channel in at least one trial has variance equal to 0')
                end
                X(ind,:) = X(ind,:) ./ repmat(sd,Tj(jj),1);
            end
        end
        covmats_unflipped_j = getCovMats(X,sum(Tj),options.maxlag,options.partial);
        if j==1
            covmats_unflipped = covmats_unflipped_j;
        else % do some kind of weighting here according to the number of samples?
            covmats_unflipped = cat(4,covmats_unflipped,covmats_unflipped_j);
        end
    end
    
else
    if isstruct(data), data = data.X; end
    if options.standardise
        for j = 1:N
            ind = (1:T(j)) + sum(T(1:j-1));
            data(ind,:) = bsxfun(@minus,data(ind,:),mean(data(ind,:)));
            sd = std(data(ind,:));
            if any(sd==0)
                error('At least one channel in at least one trial has variance equal to 0')
            end
            data(ind,:) = data(ind,:) ./ repmat(sd,T(j),1);
        end
    end
    covmats_unflipped = getCovMats(data,T,options.maxlag,options.partial);
end

end
