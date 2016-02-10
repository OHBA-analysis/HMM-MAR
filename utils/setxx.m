K = length(hmm.state);
if exist('Gamma','var')
    XXGXX = cell(K,1);
end
if ~hmm.train.multipleConf
    if ~exist('XX','var')
        XX = cell(1);
        orders = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
        if exist('data','var')
            if numel(data.X)*length(orders)>10000000
                XX{1} = single(formautoregr(data.X,T,orders,hmm.train.maxorder,hmm.train.zeromean));
            else
                XX{1} = formautoregr(data.X,T,orders,hmm.train.maxorder,hmm.train.zeromean);
            end
        else
            if numel(X)*length(orders)>10000000
                XX{1} = single(formautoregr(X,T,orders,hmm.train.maxorder,hmm.train.zeromean));
            else
                XX{1} = formautoregr(X,T,orders,hmm.train.maxorder,hmm.train.zeromean);
            end
        end
    end
    if exist('Gamma','var')
        for k=1:K
            XXGXX{k} = (XX{1}' .* repmat(Gamma(:,k)',size(XX{1},2),1)) * XX{1};
        end
    end
else
    if ~exist('XX','var')
        XX = cell(K,1);
    end
    for k=1:K
        setstateoptions;
        if isempty(XX{k})
            if exist('data','var')
                if numel(data.X)*length(orders)>10000000
                    XX{k} = single(formautoregr(data.X,T,orders,hmm.train.maxorder,train.zeromean));
                else
                    XX{k} = formautoregr(data.X,T,orders,hmm.train.maxorder,train.zeromean);
                end
            else
                if numel(X)*length(orders)>10000000
                    XX{k} = single(formautoregr(X,T,orders,hmm.train.maxorder,train.zeromean));
                else
                    XX{k} = formautoregr(X,T,orders,hmm.train.maxorder,train.zeromean);
                end
            end
        end
        if exist('Gamma','var')
            XXGXX{k} = (XX{k}' .* repmat(Gamma(:,k)',size(XX{k},2),1)) * XX{k};
        end
    end
end