function [actstates,hmm,Gamma,Xi] = getactivestates(hmm,Gamma,Xi)

%ndim = size(X,2);
K = hmm.K;
orders = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
if isfield(hmm.state(1),'Omega')
    ndim = size(hmm.state(1).Omega.Gam_rate,2);
else
    ndim = size(hmm.state(1).W.Mu_W,2);
end

Gammasum = sum(Gamma);
if isfield(hmm.train,'B'), Q = size(hmm.train.B,2);
else, Q = sum(any(hmm.train.S==1)); end

threshold = max(4*length(orders)*Q+5,10);

actstates = ones(1,K); % length = to the last no. of states (=the original K if dropstates==0)
for k=1:K
    if Gammasum(:,k) <= threshold
        if ~hmm.train.dropstates && hmm.train.active(k)==1
            if hmm.train.verbose
                fprintf('State %d has been switched off with %f points\n',k,Gammasum(k))
            end
            hmm.train.active(k) = 0;
        end
        actstates(k) = 0;
    else
        if ~hmm.train.dropstates && hmm.train.active(k)==0
            if hmm.train.verbose
                fprintf('State %d has been switched on with %f points\n',k,Gammasum(k))
            end
            hmm.train.active(k) = 1;
        end
    end
end

% if isfield(hmm.train,'grouping')
%     Q = length(unique(hmm.train.grouping));
% else
%     Q = 1;
% end
Q = 1; 
if hmm.train.dropstates == 1
    is_active = logical(actstates);
    Gamma = Gamma(:,is_active);
    Gamma = bsxfun(@rdivide,Gamma,sum(Gamma,2));
    hmm.state = hmm.state(is_active);
    if hmm.train.verbose
        knockout = find(~is_active);
        for j = 1:length(knockout)
            fprintf('State %d has been knocked out with %f points - there are %d left\n',...
                knockout(j),Gammasum(knockout(j)),K-j)
        end
    end
    hmm.K = sum(is_active);
    hmm.prior.Dir_alpha = hmm.prior.Dir_alpha(is_active);
    hmm.prior.Dir2d_alpha = hmm.prior.Dir2d_alpha(is_active,is_active);
    hmm.Dir2d_alpha = hmm.Dir2d_alpha(is_active,is_active,:);
    hmm.P = hmm.P(is_active,is_active,:);
    if Q>1
       hmm.Dir_alpha = hmm.Dir_alpha(is_active,:);
       hmm.Pi = hmm.Pi(is_active,:);
    else
       hmm.Dir_alpha = hmm.Dir_alpha(is_active);
       hmm.Pi = hmm.Pi(is_active);
    end
        
    hmm.train.Pstructure = hmm.train.Pstructure(is_active,is_active);
    hmm.train.Pistructure = hmm.train.Pistructure(is_active);
    if all(hmm.train.Pistructure==0)
       error('All states with Pistructure = true have been kicked out')
    end
    if ~isempty(Xi)
        Xi = Xi(:,is_active,is_active);
    end
    hmm.train.active = ones(1,sum(is_active));
    % Renormalize 
    for i = 1:Q
        hmm.P(:,:,i) = bsxfun(@rdivide,hmm.P(:,:,i),sum(hmm.P(:,:,i),2));
        if Q==1
            hmm.Pi = hmm.Pi ./ sum(hmm.Pi);
        else
            hmm.Pi(:,i) = hmm.Pi(:,i) ./ sum(hmm.Pi(:,i));
        end
    end

end

end
