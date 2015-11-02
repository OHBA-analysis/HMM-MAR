function [Gamma,LL] = gmm_init(data,T,options)
%
% Initialise the hidden Markov chain using an a Gaussian Mixture model
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

ndim = size(data.X,2);

if any(~isnan(data.C(:,1))),
    error('Fixing part of the Markov chain is not implemented for GMM initialisation')
end
%[~,order] = formorders(options.order,options.orderoffset,options.timelag,options.exptimelag);
order = options.maxorder;

LL = -Inf;
for n=1:options.initrep
    mix = gmm(ndim,options.K,options.covtype);
    netlaboptions = foptions;
    netlaboptions(14) = 5; % Just use 5 iterations of k-means initialisation
    mix = gmminit(mix, data.X, netlaboptions);
    netlaboptions = zeros(1, 18);
    netlaboptions(1)  = 0;                % Prints out error values.
    % Termination criteria
    netlaboptions(3) = 0.000001;          % tolerance in likelihood
    netlaboptions(14) = options.initcyc;              % Max. Number of iterations.
    % Reset cov matrix if singular values become too small
    netlaboptions(5) = 1;
    [~, netlaboutput, ~, gamma] = wgmmem(mix, data.X, ones(sum(T),1), netlaboptions);
    ll = -netlaboutput(8);     % Log likelihood of gmm model
    if options.verbose
        fprintf('Init run %d, LL %f \n',n,ll);
    end
    if ll>LL
        LL = ll;
        Gammaplus = gamma;
        s = n;
    end
end

Gamma = zeros(sum(T)-length(T)*order,options.K);
for in=1:length(T)
   t0 = sum(T(1:in-1)) - order*(in-1) + 1; t1 = sum(T(1:in)) - order*in;
   t0plus = sum(T(1:in-1)) + 1; t1plus = sum(T(1:in));
   Gamma(t0:t1,:) = Gammaplus(t0plus+order:t1plus,:);  
end

if options.verbose
    fprintf('%i-th was the best iteration with LL=%f \n',s,LL)
end

end

