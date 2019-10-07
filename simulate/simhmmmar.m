function [X,T,Gamma] = simhmmmar(T,hmm,Gamma,nrep,sim_state_tcs_only,grouping)
%
% Simulate data from the HMM-MAR
%
% INPUTS:
%
% T                     Number of time points for each time series
% hmm                   hmm structure with options specified in hmm.train
% Gamma                 State courses - leave these empty to simulate these too
% nrep                  no. repetitions of Gamma(t), from which we take the average
% sim_state_tcs_only    Flag to indicate that only state time courses will be
%                       simulated
%
% OUTPUTS
% X             simulated observations  
% T             Number of time points for each time series
% Gamma         simulated  p(state | data)
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2016)

N = length(T); K = length(hmm.state);
ndim = size(hmm.state(1).W.Mu_W,2);
if ndim==0, ndim = size(hmm.state(1).Omega.Gam_rate,1); end

if nargin<3, Gamma = []; end
if nargin<4 || isempty(nrep), nrep = 1; end
if nargin<5, sim_state_tcs_only=0; end
if nargin<6, grouping=[]; end
    

if isfield(hmm.train,'embeddedlags') && length(hmm.train.embeddedlags) > 1
    error('It is not currently possible to generate data with options.embeddedlags ~= 0'); 
end
if ~isfield(hmm.train,'order'), hmm.train.order = 0; end
if ~isfield(hmm.train,'timelag'), hmm.train.timelag = 1; end
if ~isfield(hmm.train,'exptimelag'), hmm.train.exptimelag = 1; end
if ~isfield(hmm.train,'orderoffset'), hmm.train.orderoffset = 0; end
if ~isfield(hmm.train,'S'), hmm.train.S = ones(ndim); end
if ~isfield(hmm.train,'maxorder'), hmm.train.maxorder = hmm.train.order; end

if hmm.train.maxorder > 0, d = 500;
else, d = 0;
end
nz = (~hmm.train.zeromean);

if isempty(Gamma) && K>1 % Gamma is not provided, so we simulate it too
    Gamma = simgamma(T,hmm.P,hmm.Pi,nrep,grouping);
    % Need to ensure Gamma timecourses are mutually exclusive. We generate
    % the timecourses by calculating sum_k Gamma(:, k) .* state. If not
    % discrete, we end up with "new" states that are lin. combinations of 
    % other states
    [~, maxGammaInd] = max(Gamma, [], 2);
    Gamma = zeros(size(Gamma));
    for k = 1:K, Gamma(maxGammaInd==k,k) = 1; end
    %Gamma = maxGammaInd == 1:max(maxGammaInd);
elseif isempty(Gamma) && K==1
    Gamma = ones(sum(T),1);
end
if size(Gamma,1) ~= sum(T), error('Gamma should have sum(T) rows'); end

X = zeros(sum(T),ndim);
ind = false(sum(T),1);
hmm0 = hmm; hmm0.train.zeromean = 1; 

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
    for n = 1:N
        t0 = sum(T(1:n-1)) + 1; t1 = sum(T(1:n));
        if hmm.train.maxorder > 0 % a MAR is generating the data
            if K>1
                Gamma0 = simgamma(d,hmm.P,hmm.Pi,nrep,grouping);
            else
                Gamma0 = ones(d,1);
            end
            % Ensure distinct Gamma0 time courses
            [~, maxGamma0Ind] = max(Gamma0, [], 2);
            Gamma0 = maxGamma0Ind == 1:max(maxGamma0Ind);
            X0 = simgauss(d,hmm0,Gamma0); % no mean in the innovation signal
        else % sampling Gaussian
            X0 = []; Gamma0 = [];
        end
        start = hmm.train.maxorder + 1;
        Xin = [X0; simgauss(T(n),hmm,Gamma(t0:t1,:))]; 
        if ~hmm.train.zeromean || hmm.train.maxorder > 0
            G = [Gamma0; Gamma(t0:t1,:)];
            for t=start:T(n)+d
                for k=1:K
                    orders = hmm.state(k).train.orders;
                    XX = ones(1,length(orders)*ndim + nz);
                    for i=1:length(orders)
                        o = orders(i);
                        XX(1,(1:ndim) + (i-1)*ndim + nz) = Xin(t-o,:);
                    end
                    Xin(t,:) = Xin(t,:) + G(t,k) * XX * hmm.state(k).W.Mu_W;
                end
            end
        end
        ind(t0+hmm.train.maxorder : t1) = true;
        X(t0:t1,:) = Xin(d+1:end,:);
    end
end

Gamma = Gamma(ind,:);


end

