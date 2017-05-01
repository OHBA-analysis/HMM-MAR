function data = data2struct(data,T,options)

if ~isstruct(data), data = struct('X',data); end
if ~isfield(data,'C'), 
    if options.K>1, data.C = NaN(size(data.X,1)-options.maxorder*length(T),options.K); 
    else data.C = ones(size(data.X,1),1); 
    end
elseif size(data.C,1)==sum(T) && options.maxorder>0 % we need to trim C
    ind = [];
    for j=1:length(T)
        t0 = sum(T(1:j-1)) - options.maxorder*(j-1);
        ind = [ind (t0+options.maxorder+1:t0+T(j))];
    end
    data.C = data.C(ind,:); 
    %warning('C has more rows than it should; the first rows of each trial will be discarded\n')
end

end