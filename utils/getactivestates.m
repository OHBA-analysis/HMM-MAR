function [actstates,hmm,Gamma,Xi] = getactivestates(hmm,Gamma,Xi)

%ndim = size(X,2);
K = hmm.K;
orders = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
if isfield(hmm.state(1),'Omega'), 
    ndim = size(hmm.state(1).Omega.Gam_rate,2);
else
    ndim = size(hmm.state(1).W.Mu_W,2);
end
Gammasum = sum(Gamma);

actstates = ones(1,K); % length = to the last no. of states (=the original K if dropstates==0)
for k=1:K
    if Gammasum(:,k) <= max(length(orders)*ndim+1,10)
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
    Gamma = Gamma(:,actstates==1);
    Gamma = bsxfun(@rdivide,Gamma,sum(Gamma,2));
    actstates2 = actstates;
    k = 1;
    k0 = 1;
    while k<=K
        if actstates2(k)
            k = k + 1; k0 = k0 + 1;
            continue
        end
        K = K - 1;
        hmm.state(k) = [];
        actstates2(k) = [];
        fprintf('State %d has been knocked out with %f points - there are %d left\n',k0,Gammasum(k),K)
        Gammasum(k) = [];
        k0 = k0 + 1;
    end
    hmm.K = K;
    hmm.Dir2d_alpha = hmm.Dir2d_alpha(actstates==1,actstates==1);
    hmm.Dir_alpha = hmm.Dir_alpha(actstates==1);
    hmm.prior.Dir2d_alpha = hmm.prior.Dir2d_alpha(actstates==1,actstates==1);
    hmm.prior.Dir_alpha = hmm.prior.Dir_alpha(actstates==1);
    hmm.P = hmm.P(actstates==1,actstates==1);
    hmm.Pi = hmm.Pi(actstates==1);
    Xi = Xi(:,actstates==1,actstates==1);
    hmm.train.active = ones(1,sum(actstates));
end

end
