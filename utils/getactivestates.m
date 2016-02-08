function [actstates,hmm,Gamma,Xi] = getactivestates(X,hmm,Gamma,Xi)

ndim = size(X,2);
K = hmm.K;
orders = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
Gammasum = sum(Gamma);

actstates = ones(1,K); % of length equal to the last no. of states (=the original K if dropstates==0)
for k=1:K
    if Gammasum(:,k) < max(length(orders)*ndim,5)
        if ~hmm.train.dropstates && hmm.train.active(k)==1
            fprintf('State %d has been switched off with %f points\n',k,Gammasum(k))
        end
        hmm.train.active(k) = 0;
        actstates(k) = 0;
    else
        if ~hmm.train.dropstates && hmm.train.active(k)==0
            fprintf('State %d has been switched on with %f points\n',k,Gammasum(k))
        end
        hmm.train.active(k) = 1;
    end
end

if hmm.train.dropstates==1
    Gamma = Gamma(:,actstates==1);
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
end

end
