function data = detrenddata(data,T)

for n=1:length(T)
    ind = sum(T(1:n-1))+ (1:T(n));
    if isstruct(data)
        data.X(ind,:) = detrend(data.X(ind,:));
    else
        data(ind,:) = detrend(data(ind,:));
    end
end

end
