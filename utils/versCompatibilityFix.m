function hmm = versCompatibilityFix(hmm)

K = length(hmm.state);

if ~isfield(hmm.train,'orders')
    hmm.train.orders = formorders(hmm.train.order,hmm.train.orderoffset,...
        hmm.train.timelag,hmm.train.exptimelag);
end

if isfield(hmm,'cache')
    hmm = rmfield(hmm,'cache');
end

for k=1:K
    if ~isfield(hmm.state(k),'train') || isempty(hmm.state(k).train)
        if isfield(hmm.train.state(k),'train') && ~isempty(hmm.train.state(k).train)
            hmm.state(k).train = hmm.train.state(k).train;
        else
            hmm.state(k).train = hmm.train;
        end
    end
    if isfield(hmm.state(k),'cache')
        hmm.state(k) = rmfield(hmm.state(k),'cache');
    end
    if ~isfield(hmm.state(k).train,'orders') || isempty(hmm.state(k).train.orders)
        [hmm.state(k).train.orders,hmm.state(k).train.order] = ...
            formorders(hmm.state(k).train.order,hmm.state(k).train.orderoffset,...
            hmm.state(k).train.timelag,hmm.state(k).train.exptimelag);
    end
end

if isfield(hmm.train,'state')
    hmm.train = rmfield(hmm.train,'state');
end

end