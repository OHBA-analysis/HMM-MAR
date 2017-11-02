function data = data2struct(data,T,options)

if length(options.embeddedlags)>1
    L = -min(options.embeddedlags) + max(options.embeddedlags);
else
    L = options.maxorder;
end

if ~isstruct(data), data = struct('X',data); end
if ~isfield(data,'C')
    if options.K>1, data.C = NaN(size(data.X,1)-L*length(T),options.K);
    else data.C = ones(size(data.X,1)-L*length(T),1);
    end
elseif size(data.C,1)==sum(T) && L>0 % we need to trim C
    ind = [];
    for j=1:length(T)
        t0 = sum(T(1:j-1)) - L*(j-1);
        if length(options.embeddedlags)>1
            ind = [ind (t0+1-min(options.embeddedlags)):(t0+T(j)-max(options.embeddedlags)) ];
        else
            ind = [ind (t0+options.maxorder+1:t0+T(j))];
        end
    end
    data.C = data.C(ind,:);
    %warning('C has more rows than it should; the first rows of each trial will be discarded\n')
end


end