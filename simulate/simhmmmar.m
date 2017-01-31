function [X,T,Gamma] = simhmmmar(T,hmm,Gamma,nrep,trim,X0,sim_state_tcs_only)
%
% Simulate data from the HMM-MAR
%
% INPUTS:
%
% T                     Number of time points for each time series
% hmm                   hmm structure with options specified in hmm.train
% Gamma                 State courses - leave these empty to simulate these too
% nrep                  no. repetitions of Gamma(t), from which we take the average
% trim                  how many time points to remove from the beginning of each trial
% X0                    A starting point for the time series ( no. time points x ndim x length(T) )
%                       - if not provided, it is set to Gaussian noise
% sim_state_tcs_only    Flag to indicate that only state time courses will be
%                       simulated
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
if nargin<4, nrep = 10; end
if nargin<5, trim = 0; end
if nargin<6, X0 = []; end
if nargin<7, sim_state_tcs_only=0; end

if isempty(Gamma), % Gamma is not provided, so we simulate it too
    Gamma = simgamma(T,hmm.P,hmm.Pi,nrep);
end

X = [];
if ~isfield(hmm.train,'maxorder'), hmm.train.maxorder = hmm.train.order; end

if ~sim_state_tcs_only
    for k=1:K
        if ~isfield(hmm.state(k),'train') || isempty(hmm.state(k).train)
            hmm.state(k).train = hmm.train;
        end
        if ~isfield(hmm.state(k).train,'orders')
            hmm.state(k).train.orders = ...
                formorders(hmm.state(k).train.order,...
                hmm.state(k).train.orderoffset,...
                hmm.state(k).train.timelag,...
                hmm.state(k).train.exptimelag);
        end
    end
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
end

if trim>0,
    Gamma0 = []; X0 = [];  
    for in=1:N
        t0 = sum(T(1:in-1)) + 1; t1 = sum(T(1:in));
        Gamma0 = [Gamma0; Gamma(t0+trim:t1,:)];
        if ~sim_state_tcs_only
            X0 = [X0; X(t0+trim:t1,:)];
        end
        T(in) = T(in) - trim;
    end
    Gamma = Gamma0; 
    if ~sim_state_tcs_only
        X = X0;
    end
end


end

