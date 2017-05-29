function data = detrenddata(X,T)

for n=1:length(T)
    ind = sum(T(1:n-1))+ (1:T(n));
    if isstruct(data)
        data.X(ind,:) = detrend(data.X(ind,:));
    else
        X(ind,:) = detrend(X(ind,:));
    end
end

end
