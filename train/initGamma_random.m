function Gamma = initGamma_random(T,K,D,Pstructure,Pistructure)
% Return a Gamma timecourse corresponding to random transitions with a given
% dirichletdiag (i.e. the transition probability prior)
%
% Author: Romesh Abeysuriya, University of Oxford (2017)

rng('default')
rng('shuffle') % make this "truly" random

% Form transition probability matrix
P = (D-1)*eye(K)+ones(K);
if nargin==4, P(~Pstructure) = 0; end
if nargin<5, Pistructure = true(1,K); end
P = bsxfun(@rdivide,P,sum(P,2));
Pi = zeros(1,K); 
Pi(Pistructure) = 1;
Pi = Pi / sum(Pi);

% Preallocate
Gamma = zeros(sum(T),K);

for tr=1:length(T)
    gamma = zeros(T(tr),K);
    if sum(Pi>0)==1, gamma(1,Pi>0) = 1; 
    else, gamma(1,:) = mnrnd(1,Pi); 
    end
    for t=2:T(tr)
        gamma(t,:) = mnrnd(1,P(gamma(t-1,:)==1,:));
    end
    gamma = gamma + 0.0001 * rand(T(tr),K);
    t0 = sum(T(1:tr-1)); t1 = sum(T(1:tr));
    Gamma(t0+1:t1,:) = gamma ./ repmat(sum(gamma,2),1,K);
end

end




