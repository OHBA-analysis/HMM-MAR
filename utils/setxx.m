K = length(hmm.state);
if exist('Gamma','var')
    XXGXX = cell(K,1);
end
if ~isfield(hmm.train,'multipleConf') || ~hmm.train.multipleConf
    if ~exist('B','var'), 
        if isfield(hmm.train,'B'), B = hmm.train.B;
        else B = []; end
    else
    end
    if ~exist('XX','var') || isempty(XX)
        XX = cell(1);
        orders = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
        if exist('data','var')
            if numel(data.X)*length(orders)>10000000
                XX{1} = formautoregr(data.X,T,orders,hmm.train.maxorder,hmm.train.zeromean,1,B);
            else
                XX{1} = formautoregr(data.X,T,orders,hmm.train.maxorder,hmm.train.zeromean,0,B);
            end
        else
            if numel(X)*length(orders)>10000000
                XX{1} = formautoregr(X,T,orders,hmm.train.maxorder,hmm.train.zeromean,1,B);
            else
                XX{1} = formautoregr(X,T,orders,hmm.train.maxorder,hmm.train.zeromean,0,B);
            end
        end
    end
    %if hmm.train.pcamar > 0 
    %   XXpca = zeros(size(XX{1},1),length(orders)*hmm.train.pcamar+(~zeromean));
    %   for o=1:length(orders)
    %       XXpca(:,(1:hmm.train.pcamar)+(o-1)*hmm.train.pcamar+(~zeromean)) = ...
    %        XX{1}(:,(1:ndim)+(o-1)*ndim+(~zeromean)) * hmm.train.B;
    %   end
    %   XX{1} = XXpca;
    %end
    if exist('Gamma','var')
        for k=1:K
            XXGXX{k} = (XX{1}' .* repmat(Gamma(:,k)',size(XX{1},2),1)) * XX{1};
        end
    end
    if isempty(B)
        clear B
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


