classdef (Abstract) dirichletdiags
	% Utilities to compute and test values for dirichletdiags
	%
	% COMMON TASKS
	%
	% 1. Given an expected state lifetime, what dirichletdiags should I use?
	% 		
	%	 dirichletdiags.get(0.2,100,11) <- 0.2s expected lifetime, 100Hz sampling rate, K=11
	%
	% 2. Given a value for dirichletdiags, what state lifetime does this correspond to?
	%
	%	 dirichletdiags.lifetime(190,100,11) <- dirichletdiags=190, 100Hz sampling rate, K=11
	%
	%
	% Use dotest=true to do a numerical simulation
	%
	% NB. classdef is just to create a namespace for dirichletdiags so these functions can all 
	%     stay in one file

	methods(Static)

		function [d,tested_lifetime,analytic_lifetime] = get(t,fs,K,dotest)
			% Get dirichletdiag corresponding to the mean state lifetime given a sampling rate
			% and a particular number of states
			% How many steps does t correspond to?
			% ARGUMENTS
			% t - Mean state lifetime (s)
			% fs - Data sampling rate
			% K - number of states
			if K == 1
				d = 1; % If only one state, transition matrix is 1x1 with 1 on the diagonal
				tested_lifetime = NaN;
				analytic_lifetime = NaN;
				return
			end

			if nargin < 4 || isempty(dotest) 
				dotest = false;
			end
			
			ts = fs*t; 

			f_prob = dirichletdiags.mean_lifetime(); % Function that returns the lifetime in steps given the probability
			% First, test if requested t is too small
			min_lifetime = f_prob(1/K)/fs;
			if t < min_lifetime || ts == 1
				fprintf('Smallest possible lifetime for dirichletdiag=1 -> %.2fs\n',min_lifetime)
				d = 1;
			else
				f = @(x) ts - f_prob(x); % Function with a minimum when the probability x returns a lifetime the same as requested

				[p,~,flag] = fzero(f,[1e-5 1-1e-5]); % Solve this equation numerically to find the right probability

				if flag ~= 1
					error('fzero failed')
				end

				% And convert the probability to a dirichletdiags value
				q = (1-p)/(K-1); % Off-diagonal entries
				d = p/q;
			end

			if ~dotest
				return
			end

			fprintf('Testing calculated dirichletdiag...\n')
			tested_lifetime = dirichletdiags.test_lifetime(500,100000,K,d);
			fprintf('Requested lifetime: %.3f s = %.3f steps\n',t,ts);
			fprintf('Corresponding dirichletdiag=%.2f\n',d);
			fprintf('Tested lifetime: %.2f steps = %.3f s\n',tested_lifetime,tested_lifetime/fs);
		end

		function expected_lifetime = lifetime(dirichletdiag,fs,K,dotest)
			if nargin < 4 || isempty(dotest) 
				dotest = false;
			end

			trans_p = (dirichletdiag-1)*eye(K)+ones(K);
			trans_p = bsxfun(@rdivide,trans_p,sum(trans_p,2)); % Transition probability matrix
			trans_p = trans_p(1,1); % Probability of remaining in the same state

			f = dirichletdiags.mean_lifetime();
			expected_lifetime = f(trans_p)/fs;

			if ~dotest
				return
			end

			fprintf('Testing calculated dirichletdiag...\n')
			tested_lifetime = dirichletdiags.test_lifetime(500,100000,K,dirichletdiag);
			fprintf('Dirichletdiag %.2f = %.2f steps = %.3fs\n',dirichletdiag,expected_lifetime*fs,expected_lifetime);
			fprintf('Tested lifetime: %.2f steps = %.3f s\n',tested_lifetime,tested_lifetime/fs);

		end

		function f = mean_lifetime()
			% Given a transition probability, what is the expected lifetime in units of steps?
			% Do a symbolic integration and return a function that performs this calculation and
			% returns a numeric result
			syms k_step p;
			assume(p>0);
			f =(k_step)*p^k_step*(1-p); % (probability that the state lasts k *more* steps, multiplied by lifetime which is k)
			fa = 1+int(f,k_step,0,inf); % Add 1 to the lifetime because all states last at least 1 sample
			f = @(x) double(subs(fa,p,x));
		end


		function tested_lifetime = test_lifetime(n_trials,steps_per_trial,K,dirichletdiag)
			% Run simulate_lifetimes for n_trials times, and return the average lifetime
			parfor j = 1:500
				[observed_lifetime(j)] = dirichletdiags.simulate_lifetimes(steps_per_trial,K,dirichletdiag);
			end
			tested_lifetime = mean(observed_lifetime);
		end

		function [observed_lifetime] = simulate_lifetimes(T,K,D)
			% Simulate T steps with K states and dirichletdiags D
			% Return the observed lifetime, averaged over the K states

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
		end


	end
end

