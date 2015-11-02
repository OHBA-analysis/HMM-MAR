if ~hmm.train.multipleConf
    XX = cell(1);
    [orders,order] = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
    XX{1} = formautoregr(X,T,orders,hmm.train.maxorder,hmm.train.zeromean);
    if exist('XXGXX','var')
        for k=1:K
            XXGXX{k} = (XX{1}' .* repmat(Gamma(:,k)',size(XX{1},2),1)) * XX{1};
        end
    end
else
    XX = cell(K,1);  
    for k=1:K
        setstateoptions;
        XX{k} = formautoregr(X,T,orders,hmm.train.maxorder,train.zeromean);
        if exist('XXGXX','var')
            XXGXX{k} = (XX{k}' .* repmat(Gamma(:,k)',size(XX{k},2),1)) * XX{k};
        end
    end
end