function [data_new,T_new] = downsampledata(data,T,fs_new,fs_data)

if fs_new == fs_data
    data_new = data; T_new = T; return
end

T_new = ceil((fs_new/fs_data) * T);
if isstruct(data)
    ndim = size(data.X,2); 
    data_new = struct(); 
    data_new.X = zeros(sum(T_new),ndim);
    if isfield(data,'C')
        K = size(data.C,2); N = length(T);
        L = (size(data.X,1) - size(data.C,1)) / N; 
        if K>1
            data_new.C = NaN(sum(T_new)-L*N,K);
        else
            data_new.C = ones(sum(T_new)-L*N,1);
        end
    end
else
    ndim = size(data,2); 
    data_new = zeros(sum(T_new),ndim);
end

for i=1:length(T)
    ind1 = sum(T(1:i-1))+ (1:T(i));
    ind2 = sum(T_new(1:i-1))+ (1:T_new(i));
    for n = 1:ndim
        if isstruct(data)
            data_new.X(ind2,n) = resample(data.X(ind1,n),fs_new,fs_data);
        else
            data_new(ind2,n) = resample(data(ind1,n),fs_new,fs_data);
        end
    end
end

end
