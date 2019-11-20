K = length(hmm.state);
if exist('Gamma','var') && ~isempty(Gamma)
    XXGXX = cell(K,1);
end
if ~exist('B','var')
    if isfield(hmm.train,'B'), B = hmm.train.B;
    else B = [];
    end
end
if ~exist('V','var')
    if isfield(hmm.train,'V'), V = hmm.train.V;
    else V = [];
    end
end
if ~exist('XX','var') || (size(XX,1)==0)
    orders = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
    if hmm.train.lowrank > 0
        if exist('data','var')
            XX = data.X;
        else
            XX = X;
        end
    elseif isempty(orders) && hmm.train.zeromean
        if exist('data','var')
            XX = zeros(size(data.X,1),0);
        else
            XX = zeros(size(X,1),0);
        end
    else
        if exist('data','var')
            if numel(data.X)*length(orders)>10000000
                XX = formautoregr(data.X,T,orders,hmm.train.maxorder,hmm.train.zeromean,1,B,V);
            else
                XX = formautoregr(data.X,T,orders,hmm.train.maxorder,hmm.train.zeromean,0,B,V);
            end
        else
            if numel(X)*length(orders)>10000000
                XX = formautoregr(X,T,orders,hmm.train.maxorder,hmm.train.zeromean,1,B,V);
            else
                XX = formautoregr(X,T,orders,hmm.train.maxorder,hmm.train.zeromean,0,B,V);
            end
        end
    end
end
if exist('Gamma','var') && ~isempty(Gamma)
    if ~isfield(hmm.train,'distribution') || ~strcmp(hmm.train.distribution,'logistic') || ~isfield(hmm.state,'W') 
        %do not proceed below if W not initialised yet
        for k = 1:K
            XXGXX{k} = (XX' .* repmat(Gamma(:,k)',size(XX,2),1)) * XX;
        end
    else
        n = size(XX,2)-hmm.train.logisticYdim;
        hmm = updatePsi(hmm,Gamma,XX(:,1:n),residuals);
        lambdafunc = @(psi_t) ((2*psi_t).^-1).*(log_sigmoid(psi_t)-0.5);
        for k = 1:K
            for iY=1:hmm.train.logisticYdim
                XXGXX{k,iY} = (XX(:,1:n)' .* repmat(2*lambdafunc(hmm.psi(:,iY))'.*Gamma(:,k)',n,1))* XX(:,1:n);
            end
        end
    end
end
if isempty(B), clear B; end
if isempty(V), clear V; end
