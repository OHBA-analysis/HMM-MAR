function Gamma = initGamma_random(T,K,D)
% Return a Gamma timecourse corresponding to random transitions with a given
% dirichletdiag (i.e. the transition probability prior)
%
% Author: Romesh Abeysuriya, University of Oxford (2017)

rng('default')
rng('shuffle') % make this "truly" random

% Form transition probability matrix
P = (D-1)*eye(K)+ones(K);
P = bsxfun(@rdivide,P,sum(P,2));

% Use hmmgenerate() if appropriate toolbox is installed
if license('test','statistics_toolbox')
    states = nan(sum(T),1);
    ptr = 1;
    
    for j = 1:length(T)
        [~,states(ptr:ptr+T(j)-1)] = hmmgenerate(T(j),P,eye(K));
        ptr = ptr + T(j);
    end
    
    Gamma = 0.0001*ones(sum(T),K);
    Gamma(sub2ind(size(Gamma),(1:sum(T)).',states)) = 1;
    Gamma = bsxfun(@rdivide,Gamma,sum(Gamma,2)); % Renormalize
    
else
    
    % Preallocate
    Gamma = zeros(sum(T),K);
    
    for tr=1:length(T)
        gamma = zeros(T(tr),K);
        gamma(1,:) = mnrnd(1,ones(1,K)*1/K);
        for t=2:T(tr)
            gamma(t,:) = mnrnd(1,P(gamma(t-1,:)==1,:));
        end
        gamma = gamma + 0.0001 * rand(T(tr),K);
        t0 = sum(T(1:tr-1)); t1 = sum(T(1:tr));
        Gamma(t0+1:t1,:) = gamma ./ repmat(sum(gamma,2),1,K);
    end
    
end




