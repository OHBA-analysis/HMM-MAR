train = hmm.train;
orders = train.orders;
order = max(orders); if isempty(order), order = 0; end
if ~isfield(train,'Sind') 
    if isfield(hmm.train,'V') && ~isempty(hmm.train.V)
        train.Sind = ones(size(hmm.train.V,2),hmm.train.ndim);
    else
        train.Sind = formindexes(orders,train.S);
    end
    if ~train.zeromean, train.Sind = [true(1,size(train.Sind,2)); train.Sind]; end
end
Sind = train.Sind==1; S = train.S==1;
