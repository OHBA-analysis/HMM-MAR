function [actstates,hmm,Gamma,Xi] = getactivestates(hmm,Gamma,Xi)

%ndim = size(X,2);
K = hmm.K;
orders = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
if isfield(hmm.state(1),'Omega'), 
    ndim = size(hmm.state(1).Omega.Gam_rate,2);
else
    ndim = size(hmm.state(1).W.Mu_W,2);
end
Gammasum = mean(Gamma); % Fractional occupancy

actstates = ones(1,K); % length = to the last no. of states (=the original K if dropstates==0)
for k=1:K
    if Gammasum(:,k) <= 0.01 % If state is present less than 0.01 of the time
        if ~hmm.train.dropstates && hmm.train.active(k)==1
            fprintf('State %d has been switched off with %f points\n',k,Gammasum(k))
            hmm.train.active(k) = 0;
        end
        actstates(k) = 0;
    else
        if ~hmm.train.dropstates && hmm.train.active(k)==0
            fprintf('State %d has been switched on with %f points\n',k,Gammasum(k))
            hmm.train.active(k) = 1;
        end
    end
end

if hmm.train.dropstates==1
    is_active = logical(actstates);
    Gamma = Gamma(:,is_active);
    Gamma = bsxfun(@rdivide,Gamma,sum(Gamma,2));
    hmm.state = hmm.state(is_active);

    if hmm.train.verbose
        knockout = find(~is_active);
        for j = 1:length(knockout)
            fprintf('State %d has been knocked out with %f points - there are %d left\n',knockout(j),Gammasum(knockout(j)),K)
        end
    end

    hmm.K = sum(is_active);
    hmm.Dir2d_alpha = hmm.Dir2d_alpha(is_active,is_active);
    hmm.Dir_alpha = hmm.Dir_alpha(is_active);
    hmm.prior.Dir2d_alpha = hmm.prior.Dir2d_alpha(is_active,is_active);
    hmm.prior.Dir_alpha = hmm.prior.Dir_alpha(is_active);
    hmm.P = hmm.P(is_active,is_active);
    hmm.Pi = hmm.Pi(is_active);
    Xi = Xi(:,is_active,is_active);
    hmm.train.active = ones(1,sum(is_active));
end

end
