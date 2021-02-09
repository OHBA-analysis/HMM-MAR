K = ness.K;
if ~exist('XX','var') || (size(XX,1)==0)
    orders = formorders(ness.train.order,ness.train.orderoffset,ness.train.timelag,ness.train.exptimelag);
    if ness.train.lowrank > 0
        if exist('data','var')
            XX = data.X;
        else
            XX = X;
        end
    elseif isempty(orders) && ness.train.zeromean
        if exist('data','var')
            XX = zeros(size(data.X,1),0);
        else
            XX = zeros(size(X,1),0);
        end
    else
        if exist('data','var')
            if numel(data.X)*length(orders)>10000000
                XX = formautoregr(data.X,T,orders,ness.train.maxorder,ness.train.zeromean,1);
            else
                XX = formautoregr(data.X,T,orders,ness.train.maxorder,ness.train.zeromean,0);
            end
        else
            if numel(X)*length(orders)>10000000
                XX = formautoregr(X,T,orders,ness.train.maxorder,ness.train.zeromean,1);
            else
                XX = formautoregr(X,T,orders,ness.train.maxorder,ness.train.zeromean,0);
            end
        end
    end
end
