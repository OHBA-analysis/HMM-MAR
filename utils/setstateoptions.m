if isfield(hmm.state(k),'train') && ~isempty(hmm.state(k).train), train = hmm.state(k).train;
else train = hmm.train;
end
[orders,order] = formorders(train.order,train.orderoffset,train.timelag,train.exptimelag);
Sind = train.Sind==1; S = train.S==1;
if hmm.train.multipleConf, kk = k;
else kk = 1;
end