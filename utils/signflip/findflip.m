function [flips,scorepath,covmats_unflipped] = findflip(X,T,options)
% Finds an optimal permutation of the channels, where goodness is measured
% as the mean lagged partial  cross-correlation across pair of channels and lags.
% In other words, it finds a permutation where the signs of the lagged
% partial correlations agree as much as possible across subjects.
%
% INPUTS
% X             time series, or alternatively an (unflipped) array of
%                   autocorrelation matrices (ndim x ndim x no.lags x no. trials),
%                   as computed for example by getCovMats()
% T             length of series
% options:
%  maxlag        max lag to consider 
%  nbatch        no. of channels to evaluate at each iteration
%  noruns        how many random initialisations will be carried out
%  standardise   if 1, standardise the data
%  partial       if 1, base on partial correlation instead of correlation 
%  maxcyc        for each initialization, maximum number of cycles of the greedy algorithm
%  mincyc        for each initialization, minimum number of cycles of the greedy algorithm
%  verbose       do we get loud?
%
% OUTPUT
% flips         (length(T) X No. channels) binary matrix saying which channels must be 
%               flipped for each time series
% scorepath     cell with the score of the winning solutions
% covmats_unflipped  the disambiguated covariance matrices
%
% Author: Diego Vidaurre, University of Oxford.

N = length(T); ndim = size(X,2);

if ~isfield(options,'maxlag'), options.maxlag = 10; end
if ~isfield(options,'noruns'), options.noruns = 50; end
if ~isfield(options,'maxcyc'), options.maxcyc = 100*N*ndim; end
if ~isfield(options,'mincyc'), options.mincyc = 10; end
if ~isfield(options,'probinitflip'), options.probinitflip = 0.25; end
if ~isfield(options,'nbatch'), options.nbatch = ndim; end
if ~isfield(options,'standardise'), options.standardise = 1; end
if ~isfield(options,'partial'), options.partial = 0; end
if ~isfield(options,'verbose'), options.verbose = 1; end

if options.maxcyc<options.mincyc
    error('maxcyc cannot be lower than mincyc')
end

if length(size(X))==4 % it is an array of autocorrelation matrices already
    covmats_unflipped = X; clear X
else
    if options.standardise
        for n = 1:N
            ind = (1:T(n)) + sum(T(1:n-1));
            X(ind,:) = X(ind,:) - repmat(mean(X(ind,:)),T(n),1);
            sd = std(X(ind,:));
            if any(sd==0) 
                error('At least one channel in at least one trial has variance equal to 0')
            end
            X(ind,:) = X(ind,:) ./ repmat(sd,T(n),1);
        end
    end
    covmats_unflipped = getCovMats(X,T,options.maxlag,options.partial);
end

score = -Inf;
scorepath = cell(options.noruns,1);

for r = 1:options.noruns
    
    if ~isfield(options,'Flips')
        if r==1 % first run starts at no flips
            flipsr = zeros(N,ndim);
        else
            flipsr = binornd(1,options.probinitflip,N,ndim); % random init
        end
    else
        flipsr = options.Flips;
    end
    
    for cyc=1:options.maxcyc
        if cyc==1 || ch
            signmats = getSignMat(flipsr);
            covmats = applySign(covmats_unflipped,signmats);
            [scorer,scorers] = getscore(covmats);
        end
        ScoreMatrix = zeros(N,ndim);
        channels = randperm(ndim,options.nbatch);
        if cyc==1  
            scorepath{r} = scorer;
            if options.verbose
                fprintf('Run %d, Init, score %f \n',r,scorer)
            end
        else
            scorepath{r} = [scorepath{r} scorer];
        end
        for d = channels
            for n = 1:N
                cm = covmats(:,:,:,n); 
                sm = signmats(:,:,n);
                flipsr(n,d) = ~flipsr(n,d); % do ...
                signmats(:,:,n) = getSignMat(flipsr(n,:)); 
                covmats(:,:,:,n) = applySign(covmats_unflipped(:,:,:,n),signmats(:,:,n));
                s = getscore(covmats,n,scorers); % ... evaluate ...
                covmats(:,:,:,n) = cm;
                signmats(:,:,n) = sm;
                flipsr(n,d) = ~flipsr(n,d); % ... and undo
                ScoreMatrix(n,d) = s - scorer;
            end
        end
        [score1,I] = max(ScoreMatrix(:));
        [n,d] = ind2sub([N ndim],I);
        if score1>0
            flipsr(n,d) = ~flipsr(n,d);
            if options.verbose
                fprintf('Run %d, Cycle %d, score +%f, flipped (%d,%d) \n',r,cyc,score1,n,d)
            end
            if cyc==options.maxcyc % we are finishing
                scorer = scorer+score1;
                scorepath{r} = [scorepath{r} scorer];
            else
                ch = 1;
            end
        elseif cyc>=options.mincyc % we are finishing
            if options.verbose
                fprintf('Convergence, score=%f \n',scorer)
            end
            scorepath{r} = [scorepath{r} scorer];
            break
        else
            ch = 0;
            if options.verbose
                fprintf('Run %d, Cycle %d, score +0\n',r,cyc)
            end
        end
    end
    
    if scorer>score
        score = scorer;
        flips = flipsr;
    end
   
end

if options.verbose
    fprintf('Final Score=%f\n',score)
end

% Among the equivalent flippings, we keep the one w/ the lowest no. of flips
for n = 1:N
    if mean(flips(n,:))>0.5
        flips(n,:) = 1 - flips(n,:);
    end
end

if nargout>1
    for n = 1:N
        ind = (1:T(n)) + sum(T(1:n-1));
        for d = 1:ndim
            if flips(n,d)==1
                X(ind,d) = -X(ind,d);
            end
        end
    end
end

end











