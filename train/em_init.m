function [Gamma,LL] = em_init(data,T,op,Sind)
%
% Initialise the hidden Markov chain using an EM algorithm for clusterwise regression in time series.
% (It uses the default state configuration for all states)
%
% INPUT
% data      observations, a struct with X (time series) and C (classes, optional)
% T         length of observation sequence
% options,  structure with the training options - different from HMMMAR are
%   nu            initialisation parameter; default T/200
%   initrep     maximum number of repetitions
%   initcyc     maximum number of iterations; default 100
%
% OUTPUT
% Gamma     p(state given X)
% LL        the final model log-likelihood
%
% Author: Diego Vidaurre, University of Oxford

orders = formorders(op.order,op.orderoffset,op.timelag,op.exptimelag);

N = length(T);
Gamma = [];

Y = []; C = [];
for in=1:N
    t0 = sum(T(1:in-1));
    Y = [Y; data.X(t0+1+op.maxorder:t0+T(in),:)];
    C = [C; data.C(t0+1+op.maxorder:t0+T(in),:)];
end

XX = formautoregr(data.X,T,orders,op.maxorder,op.zeromean);
if ~op.zeromean, Sind = [true(1,size(Sind,2)); Sind]; end

LL = -Inf; 
for n=1:op.initrep
    while 1
        [gamma,ll] = emclrgr(XX,Y,C,Sind==1,op.K,op.nu,op.tol,op.initcyc,0);
        %if order>0, [gamma,ll] = emclrgr(XX,Y,K,nu,diagcov,tol,maxit,0);
        %else [gamma,ll] = emmixg(X,K,nu,diagcov,tol,maxit,0);
        %end
        if (~isinf(ll)), break;
        else fprintf('Overflow \n');
        end
    end
        
    if op.verbose
        fprintf('Init run %d, LL %f \n',n,ll);
    end
    if ll>LL
        LL = ll;
        Gamma = gamma;
        s = n;
    end
end
if op.verbose
    fprintf('%i-th was the best iteration with LL=%f \n',s,LL)
end

end

