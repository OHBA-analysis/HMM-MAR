function d = compute_dirichletdiag(t,fs,K,dotest)
	% Get dirichletdiag corresponding to the mean state lifetime given a sampling rate
	% and a particular number of states
	% How many steps does t correspond to?
	% ARGUMENTS
	% t - Mean state lifetime (s)
	% fs - Data sampling rate
	% K - number of states

	if nargin < 4 || isempty(dotest) 
		dotest = false;
	end
	
	ts = round(fs*t);
	ts = max(1,ts); % Must be at least 1 step

	% Given transition probability, calculate state lifetime and minimize the difference
	f = @(p) ts - (1 + -(p-1)*p/((log(p)^2)));

	[p,~,flag] = fzero(f,[eps 1-eps]);
	if flag ~= 1
		error('fzero failed')
	end

	% And d is the corresponding dirichletdiag
	q = (1-p)/(K-1); % Off-diagonal entries
	d = max(1,round(p/q));

	if ~dotest
		return
	end

	fprintf('Testing calculated dirichletdiag...\n')
	parfor j = 1:500
		[observed_lifetime(j),analytic_lifetime] = hammer_utils.test_lifetime(100000,K,d);
	end
	fprintf('Requested lifetime: %.2f s = %d steps\n',t,ts);
	fprintf('Corresponding dirichletdiag=%d\n',d);
	fprintf('Tested lifetime: %.2f steps = %.2f s\n',mean(observed_lifetime),mean(observed_lifetime)/fs);

function [observed_lifetime,analytic_lifetime] = test_lifetime(T,K,D)

	use_hmmgenerate = 1; % Should give same result, but about 100x faster

	if use_hmmgenerate
	    P = (D-1)*eye(K)+ones(K);
	    P = bsxfun(@rdivide,P,sum(P,2));
	    [~,states] = hmmgenerate(T,P,eye(K));
	    Gamma = 0.0001*ones(T,K);
	    Gamma(sub2ind(size(Gamma),1:T,states)) = 1;
	    Gamma = bsxfun(@rdivide,Gamma,sum(Gamma,2)); % Renormalize
	else
	    P = ones(K,K); 
	    P = P - diag(diag(P)) + D*eye(K);
	    for k=1:K
	        P(k,:)=P(k,:)./sum(P(k,:),2);
	    end
	    % Copied from initGamma_random
	    Gamma = initGamma_random(T,K,D);
	end

	for j = 1:K
	    G = round(Gamma(:,j));
	    t = find(diff(G));
	    if G(1)
	        t_off = t(1:2:end);
	        t_on = t(2:2:end);
	    else
	        t_off = t(2:2:end);
	        t_on = t(1:2:end);
	    end

	    if t_on(1) > t_off(1)
	        t_off = t_off(2:end);
	    end

	    if t_on(end) > t_off(end)
	        t_on = t_on(1:end-1);
	    end

	    lifetime = t_off - t_on;
	    l(j) = mean(lifetime(j));
	    
	end
	observed_lifetime = mean(l);
	p = P(1,1);
	analytic_lifetime = 1 + -(p-1)*p/((log(p)^2));

