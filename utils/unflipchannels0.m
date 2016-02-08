function [Flips,score,X] = unflipchannels(X,T,options)
% Finds an optimal permutation of the channels, where goodness is measured
% as the mean lagged partial  cross-correlation across pair of channels and lags.
% In other words, it finds a permutation where the signs of the lagged
% partial correlations agree as much as possible across subjects.
%
% INPUTS
% X             time series
% T             length of series
% options:
%  maxlag        max lag to consider 
%  nbatch        no. of channels to look at at each iteration
%  randominit    how many random initializations we're doing
%  standardize   if 1, standardize the data
%  maxcyc        for each initialization, maximum number of cycles of the greedy algorithm
%
% OUTPUT
% Flips         No. of time series X No. channels binary matrix saying which channels must be flipped for each time series
% score         score of the winning solutions
%
% Author: Diego Vidaurre, University of Oxford.

N = length(T); ndim = size(X,2);

if ~isfield(options,'maxlag'), options.maxlag = 4; end
if ~isfield(options,'randominit'), options.randominit = 10; end
if ~isfield(options,'maxcyc'), options.maxcyc = 100*N*ndim; end
if ~isfield(options,'probinitflip'), options.probinitflip = 0.25; end
if ~isfield(options,'nbatch'), options.nbatch = ndim; end
if ~isfield(options,'standardize'), options.standardize = 1; end
if ~isfield(options,'verbose'), options.verbose = 1; end

if options.standardize
    for in = 1:N
        ind = (1:T(in)) + sum(T(1:in-1));
        X(ind,:) = X(ind,:) - repmat(mean(X(ind,:)),T(in),1);
        X(ind,:) = X(ind,:) ./ repmat(std(X(ind,:)),T(in),1);
    end
end

score = -Inf;

for r = 1:options.randominit
    
    if ~isfield(options,'Flips')
        Flipsr = binornd(1,options.probinitflip,N,ndim); % random init
    else
        Flipsr = options.Flips;
    end
    iCV = getCovMats(X,T,options.maxlag,Flipsr);
    scorer = getscore(iCV);
    if options.verbose
        fprintf('Run %d, Init, score %f \n',r,scorer)
    end
    for cyc=1:options.maxcyc
        ScoreMatrix = zeros(N,ndim);
        channels = randperm(ndim,options.nbatch);
        for d = channels
            for in = 1:N
                Flipsr(in,d) = ~Flipsr(in,d);
                ind = (1:T(in)) + sum(T(1:in-1));
                icv = iCV(:,:,:,in);
                s0 = getscore(iCV);
                iCV(:,:,:,in) = getCovMats(X(ind,:),T(in),options.maxlag,Flipsr(in,:));
                s1 = getscore(iCV);
                iCV(:,:,:,in) = icv;
                ScoreMatrix(in,d) = s1 - s0;
                Flipsr(in,d) = ~Flipsr(in,d);
            end
        end
        [score1,I] = max(ScoreMatrix(:));
        [in,d] = ind2sub([N ndim],I);
        ind = (1:T(in)) + sum(T(1:in-1));
        if score1>0
            scorer = score1;
            Flipsr(in,d) = ~Flipsr(in,d);
            iCV(:,:,:,in) = getCovMats(X(ind,:),T(in),options.maxlag,Flipsr(in,:));
            if options.verbose
                fprintf('Run %d, Cycle %d, score %f, flipped (%d,%d) \n',r,cyc,scorer,in,d)
            end
        else
            if options.verbose
                fprintf('Convergence %f \n',scorer)
            end
            break
        end
    end
    
    if scorer>score,
        score = scorer;
        Flips = Flipsr;
    end
    
    for in = 1:N
        if mean(Flips(in,:))>0.5
            Flips(in,:) = 1 - Flips(in,:);
        end
    end
    
end

if nargout==3,
    for in = 1:N
        ind = (1:T(in)) + sum(T(1:in-1));
        for d = 1:ndim
            if Flipsr(in,d)
                X(ind,d) = -X(ind,d);
            end
        end
    end
end

end


% function iCovMats = getinvCovMats(X,T,maxlag,Flips)
% N = length(T); ndim = size(X,2);
% iCovMats = zeros(ndim,ndim,maxlag+1,N);
% t0 = 0;
% for in=1:N 
%     t1 = t0 + T(in);
%     Xin = X(t0+1:t1,:); t0 = t1;
%     for i=1:ndim, if Flips(in,i)==1, Xin(:,i) = -Xin(:,i); end; end
%     for j = 0:maxlag
%         iCovMats(:,:,j+1,in) = inv(Xin(1:T(in)-max(lags),:)' * Xin(1+j:T(in)-max(lags)+j,:));
%     end    
% end
% end




