function Gamma = initGamma_random(T,K,D,Pstructure,Pistructure,nessmodel,priorOFFvsON)
% Return a Gamma timecourse corresponding to random transitions with a given
% dirichletdiag (i.e. the transition probability prior)
% For NESS , D must be a 2x2 matrix of prior transitions - or left
%   empty
%   first state is ON, second is OFF  (always start in OFF)
%
% Author: Romesh Abeysuriya, University of Oxford (2017)
%         Diego Vidaurre, Aarhus University (2020) 

if nargin < 4, Pstructure = []; end 
if nargin < 5, Pistructure = []; end 
if nargin < 6, nessmodel = false; end 
if nargin < 7, priorOFFvsON = 3; end 

rng('default')
rng('shuffle') % make this "truly" random

% Form transition probability matrix
if nessmodel
    P = (D-1)*eye(2)+ones(2);
    P(:,2) = P(:,2) * priorOFFvsON;
    P = bsxfun(@rdivide,P,sum(P,2));
    Pi = [0 1];
    if ~isempty(Pstructure) || ~isempty(Pistructure)
       warning('initGamma_random: Pstructure and Postructure will be ignored') 
    end
else
    P = (D-1)*eye(K)+ones(K);
    if nargin>=4, P(~Pstructure) = 0; end
    if nargin<5, Pistructure = true(1,K); end
    P = bsxfun(@rdivide,P,sum(P,2));
    Pi = zeros(1,K);
    Pi(Pistructure) = 1;
    Pi = Pi / sum(Pi);
end

% Preallocate
Gamma = zeros(sum(T),K);

if nessmodel
    for n = 1:length(T)
        for k = 1:K
            while true
                gamma = zeros(T(n),2);
                gamma(1,2) = 1;
                for t = 2:T(n)
                    gamma(t,:) = mnrnd(1,P(gamma(t-1,:)==1,:));
                end
                if all(sum(gamma)>0), break; end
            end
            gamma = gamma + 0.0001 * rand(T(n),2);
            gamma = gamma ./ repmat(sum(gamma,2),1,2);
            t0 = sum(T(1:n-1)); t1 = sum(T(1:n));
            Gamma(t0+1:t1,k) = gamma(:,1); 
        end
    end
elseif isChain(P) % If it is a chain, it P must be ordered
    for n = 1:length(T)
        ch = cumsum(rand(1,K));
        ch = ch / ch(end);
        ch = round(T(n) * ch);
        ch = [0 ch(1:end-1)];
        gamma = zeros(T(n),K);
        for k = 1:K-1, gamma(ch(k)+1:ch(k+1),k) = 1; end
        gamma(ch(k+1)+1:end,K) = 1;
        t0 = sum(T(1:n-1)); t1 = sum(T(1:n));
        Gamma(t0+1:t1,:) = gamma;
    end
else
    for n = 1:length(T)
        gamma = zeros(T(n),K);
        if sum(Pi>0)==1, gamma(1,Pi>0) = 1;
        else, gamma(1,:) = mnrnd(1,Pi);
        end
        for t = 2:T(n)
            gamma(t,:) = mnrnd(1,P(gamma(t-1,:)==1,:));
        end
        gamma = gamma + 0.0001 * rand(T(n),K);
        t0 = sum(T(1:n-1)); t1 = sum(T(1:n));
        Gamma(t0+1:t1,:) = gamma ./ repmat(sum(gamma,2),1,K);
    end
end

end


function bool = isChain(P)
%bool = any(sum(P,2)==1);
P = P~=0;
bool = all(sum(P(1:end-1,:),2)==2) && sum(P(end,:))==1;
end

