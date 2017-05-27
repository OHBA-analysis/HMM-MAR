function hmm = obsinference(data,T,Gamma,hmm,residuals,XX)

N = length(T);
K = length(hmm.state);
order = hmm.train.maxorder;

if ~isstruct(data)
    data = struct('X',data);
    data.C = NaN(size(data.X,1)-order*length(T),K);
end

% Compute residuals
if nargin < 5 || isempty(residuals)
    ndim = size(data.X,2);
    if ~isfield(hmm.train,'Sind')
        orders = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
        hmm.train.Sind = formindexes(orders,hmm.train.S);
    end
    if ~hmm.train.zeromean, hmm.train.Sind = [true(1,ndim); hmm.train.Sind]; end
    residuals =  getresiduals(data.X,T,hmm.train.Sind,hmm.train.maxorder,hmm.train.order,...
        hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag,hmm.train.zeromean);
end

% Compute XX, XXGXX
if nargin < 6 || isempty(XX)
    setxx;
end


hmm = obsupdate(T,Gamma,hmm,residuals,XX,XXGXX);

end