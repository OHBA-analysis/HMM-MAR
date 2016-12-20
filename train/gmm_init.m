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
ll = nan(1,options.initrep);
gamma = cell(1,options.initrep);

if options.useParallel % not very elegant
parfor n=1:options.initrep
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
    [~, netlaboutput, ~, gamma{n}] = wgmmem(mix, data.X, ones(sum(T),1), netlaboptions);
    ll(n) = -netlaboutput(8);     % Log likelihood of gmm model
    if options.verbose
        fprintf('Init run %d, LL %f \n',n,ll(n));
    end
end
else
for n=1:options.initrep
    mix = gmm(ndim,options.K,options.covtype);
    netlaboptions = foptions;
    netlaboptions(14) = 5; % Just use 5 iterations of k-means initialisation
    try
        mix = gmminit(mix, data.X, netlaboptions);
    catch excp,
        disp('Probably, part of your time series has no information and can be removed.')
        disp('For example, check if there are segments such that data(during_segment,:) == 0')
        error('Error computing the inverse of the covariance matrix.')
    end
    netlaboptions = zeros(1, 18);
    netlaboptions(1)  = 0;                % Prints out error values.
    % Termination criteria
    netlaboptions(3) = 0.000001;          % tolerance in likelihood
    netlaboptions(14) = options.initcyc;              % Max. Number of iterations.
    % Reset cov matrix if singular values become too small
    netlaboptions(5) = 1;
    [~, netlaboutput, ~, gamma{n}] = wgmmem(mix, data.X, ones(sum(T),1), netlaboptions);
    ll(n) = -netlaboutput(8);     % Log likelihood of gmm model
    if options.verbose
        fprintf('Init run %d, LL %f \n',n,ll(n));
    end
end    
end

[~,s] = max(ll);
Gammaplus = gamma{s};
LL = ll(s);

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

function OPTIONS=foptions(parain)
%FOPTIONS Default parameters for use with SIMCNSTR.
%
%   OPTIONS = FOPTIONS returns the default options.
%
%   OPTIONS = FOPTIONS(SOME_OPTIONS) takes the non-empty vector 
%     SOME_OPTIONS and replaces the zero or missing values of 
%     SOME_OPTIONS with the default options.
%
%   The parameters are:
%   OPTIONS(1)-Display parameter (Default:0). 1 displays some results.
%   OPTIONS(2)-Termination tolerance for X.(Default: 1e-4).
%   OPTIONS(3)-Termination tolerance on F.(Default: 1e-4).
%   OPTIONS(4)-Termination criterion on constraint violation.(Default: 1e-6)
%   OPTIONS(5)-Algorithm: Strategy:  Not always used.
%   OPTIONS(6)-Algorithm: Optimizer: Not always used. 
%   OPTIONS(7)-Algorithm: Line Search Algorithm. (Default 0)
%   OPTIONS(8)-Function value. (Lambda in goal attainment. )
%   OPTIONS(9)-Set to 1 if you want to check user-supplied gradients
%   OPTIONS(10)-Number of Function and Constraint Evaluations.
%   OPTIONS(11)-Number of Function Gradient Evaluations.
%   OPTIONS(12)-Number of Constraint Evaluations.
%   OPTIONS(13)-Number of equality constraints. 
%   OPTIONS(14)-Maximum number of function evaluations. 
%               (Default is 100*number of variables)
%   OPTIONS(15)-Used in goal attainment for special objectives. 
%   OPTIONS(16)-Minimum change in variables for finite difference gradients.
%   OPTIONS(17)-Maximum change in variables for finite difference gradients.
%   OPTIONS(18)-Step length. (Default 1 or less). 

%   Copyright 1990-2003 The MathWorks, Inc. 
%   $Revision: 1.1.6.1 $  $Date: 2003/12/31 19:52:53 $

if nargin<1; 
   parain = [];
 end
sizep=length(parain);
OPTIONS=zeros(1,18);
OPTIONS(1:sizep)=parain(1:sizep);
default_options=[0,1e-4,1e-4,1e-6,0,0,0,0,0,0,0,0,0,0,0,1e-8,0.1,0];
OPTIONS=OPTIONS+(OPTIONS==0).*default_options;
end
