function data = standardisedata(data,T,standardise,valid_dims)

N = length(T);

if standardise
    for i = 1:N
        t = (1:T(i)) + sum(T(1:i-1));
        if isstruct(data)
            data.X(t,:) = bsxfun(@minus,data.X(t,:),mean(data.X(t,:))); 
            sdx = std(data.X(t,:));
            if any(sdx==0)
                error('At least one of the trials/segments/subjects has variance equal to zero (use cleandata4hmm?)');
            end
            data.X(t,:) = bsxfun(@rdivide,data.X(t,:),sdx); 
        else
            data(t,:) = bsxfun(@minus,data(t,:),mean(data(t,:)));  
            sdx = std(data(t,:));
            if any(sdx==0)
                error('At least one of the trials/segments/subjects has variance equal to zero (use cleandata4hmm?)');
            end
            data(t,:) = bsxfun(@rdivide,data(t,:),sdx);
        end
    end
else 
    if nargin<4 % this is to avoid the function complaining when TUDA is used
        if isstruct(data)
            valid_dims = [1:size(data.X,2)];
        else
            valid_dims = [1:size(data,2)];
        end
    end
    for i = 1:N
        t = (1:T(i)) + sum(T(1:i-1));
        if isstruct(data)
            if any(std(data.X(t,valid_dims))==0)
                error('At least one of the trials/segments/subjects has variance equal to zero (use cleandata4hmm?)');
            end
        else
            if any(std(data(t,valid_dims))==0)
                error('At least one of the trials/segments/subjects has variance equal to zero (use cleandata4hmm?)');
            end
        end
    end
end

end