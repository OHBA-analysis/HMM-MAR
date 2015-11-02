function [X,T,Gamma]=simhmmmar(T,hmm,Gamma,trim,X0)
%
% Simulate data from the HMM-MAR
%
% INPUTS:
%
% T             Number of time points for each time series
% hmm           hmm structure with options specified in hmm.train
% Gamma         Initial state courses
% trim          how many time points to remove from the beginning of each trial
% X0            A starting point for the time series ( no. time points x ndim x length(T) )
%                   - if not provided, it is set to Gaussian noise
%
% OUTPUTS
% X             simulated observations  
% T             Number of time points for each time series
% Gamma         simulated  p(state | data)
%
% Author: Diego Vidaurre, OHBA, University of Oxford

N = length(T); K = length(hmm.state);
ndim = size(hmm.state(1).W.Mu_W,2); 
span = .1;  

if nargin<3, Gamma = []; end
if nargin<4, trim = 0; end
if nargin<5, X0 = []; end


if isempty(Gamma), % Gamma is not provided, so we simulate it too
    for in=1:N
        Gammai = zeros(T(in),K);
        Gammai(1,:) = mnrnd(1,hmm.Pi);
        for t=2:T(in)
            Gammai(t,:) = mnrnd(1,hmm.P(Gammai(t-1,:)==1,:));
        end
        for k=1:K, 
            Gammai(:,k) = smooth(Gammai(:,k),'lowess',span); 
        end
        Gamma = [ Gamma;  Gammai ./ repmat(sum(Gammai,2),1,K) ];
    end
end

X = [];

for in=1:N
    t0 = sum(T(1:in-1)) + 1; t1 = sum(T(1:in));
    if isempty(X0)
        Xin = simgauss(T(in),hmm,Gamma(t0:t1,:));
        start = hmm.train.maxorder + 1; 
    else
        Xin = zeros(T(in),ndim);
        start = size(X0,1);
        Xin(1:start,:) = X0(:,:,in);
        Xin(start+1:end,:) = simgauss(T(in)-start,hmm,Gamma(t0+start:t1,:));
        start = start + 1; 
    end
    for t=start:T(in)
        for k=1:K
            setstateoptions;
            XX = zeros(1,length(orders)*ndim);
            for i=1:length(orders)
                o = orders(i);
                XX(1,(1:ndim) + (i-1)*ndim) = Xin(t-o,:);
            end;
            if ~hmm.train.zeromean
                XX = [1 XX];
            end
            Xin(t,:) = Xin(t,:) + Gamma(t,k) * XX * hmm.state(k).W.Mu_W;
        end
    end   
    X = [X; Xin];
end

if trim>0,
    Gamma0 = []; X0 = [];
    for in=1:N
        t0 = sum(T(1:in-1)) + 1; t1 = sum(T(1:in));
        Gamma0 = [Gamma0; Gamma(t0+trim:t1,:)];
        X0 = [X0; X(t0+trim:t1,:)];
        T(in) = T(in) - trim;
    end
    Gamma = Gamma0; X = X0;
end


end


function X = simgauss(T,hmm,Gamma)

ndim = size(hmm.state(1).W.Mu_W,2); K = size(Gamma,2);
X = zeros(T,ndim);
mu = zeros(T,ndim);

switch hmm.train.covtype
    case 'uniquediag'
        Cov = hmm.Omega.Gam_rate / hmm.Omega.Gam_shape;
        X = repmat(Cov,T,1) .* randn(T,ndim);
    case 'diag'
        for k=1:K
            Cov = hmm.state(k).Omega.Gam_rate / hmm.state(k).Omega.Gam_shape;
            X = X + repmat(Gamma(:,k),1,ndim) .* repmat(Cov,T,1) .* randn(T,ndim);
        end
    case 'uniquefull'
        Cov = hmm.Omega.Gam_rate / hmm.Omega.Gam_shape;
        X = mvnrnd(mu,Cov);
    case 'full'
        for k=1:K
            Cov = hmm.state(k).Omega.Gam_rate / hmm.state(k).Omega.Gam_shape;
            X = X + repmat(Gamma(:,k),1,ndim) .* mvnrnd(mu,Cov);
        end        
end

end



