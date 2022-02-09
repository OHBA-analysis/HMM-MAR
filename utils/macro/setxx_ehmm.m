K = ehmm.K;
if ~exist('XX','var') || (size(XX,1)==0)
    orders = formorders(ehmm.train.order,ehmm.train.orderoffset,ehmm.train.timelag,ehmm.train.exptimelag);
    if ehmm.train.lowrank > 0
        if exist('data','var')
            XX = data.X;
        else
            XX = X;
        end
    elseif isempty(orders) && ehmm.train.zeromean
        if exist('data','var')
            XX = zeros(size(data.X,1),0);
        else
            XX = zeros(size(X,1),0);
        end
    else
        if exist('data','var')
            if numel(data.X)*length(orders)>10000000
                XX = formautoregr(data.X,T,orders,ehmm.train.maxorder,ehmm.train.zeromean,1);
            else
                XX = formautoregr(data.X,T,orders,ehmm.train.maxorder,ehmm.train.zeromean,0);
            end
        else
            if numel(X)*length(orders)>10000000
                XX = formautoregr(X,T,orders,ehmm.train.maxorder,ehmm.train.zeromean,1);
            else
                XX = formautoregr(X,T,orders,ehmm.train.maxorder,ehmm.train.zeromean,0);
            end
        end
    end
end
