function [flips,scorepath,covmats_unflipped] = findflip(data,T,options,data_ref,T_ref)
% Finds an optimal permutation of the channels, where goodness is measured
% as the mean lagged partial  cross-correlation across pair of channels and lags.
% In other words, it finds a permutation where the signs of the lagged
% partial correlations agree as much as possible across subjects.
% If 'data_ref' is specified, then it finds the optimal permutation of the channels
% for 'data' such that the mean lagged partial  cross-correlation across 
% pair of channels and lags matches as much as possible that of
% 'data_ref', which is assumed to have been sign-corrected already.
% This is useful to compare results between runs and data sets. 
%
% INPUTS
% data          observations, either a struct with X (time series) 
%                             or just a matrix containing the time series
%               Alternatively, it can be an (unflipped) array of
%                   autocorrelation matrices (ndim x ndim x no.lags x no. trials),
%                   as computed for example by getCovMats()
% T             length of series. If data is supplied as the
%               autocorrelation matrices, then this is not required and 
%               can be specified as []
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
% data_ref and T_ref refer to having a previous data set to
% with which we want to be consistent (see above). These are assumed to be
% already sign-disambiguated 

% OUTPUT
% flips         (length(T) X No. channels) binary matrix saying which channels must be 
%               flipped for each time series
% scorepath     cell with the score of the winning solutions
% covmats_unflipped  the disambiguated covariance matrices
%
% Author: Diego Vidaurre, University of Oxford.

N = length(T); 

if nargin < 3, options = struct; end
if ~isfield(options,'maxlag'), options.maxlag = 10; end
if ~isfield(options,'noruns'), options.noruns = 10; end
if ~isfield(options,'mincyc'), options.mincyc = 10; end
if ~isfield(options,'probinitflip'), options.probinitflip = 0.25; end
if ~isfield(options,'standardise'), options.standardise = 1; end
if ~isfield(options,'partial'), options.partial = 0; end
if ~isfield(options,'verbose'), options.verbose = 1; end

if length(size(data))==4 % it is an array of autocorrelation matrices already
    covmats_unflipped = data; clear data
    ndim = size(covmats_unflipped,1);
else
    covmats_unflipped = getAllCovMats(data,T,options);
    ndim = size(covmats_unflipped,1);
end
    
if nargin>3
    if length(size(data_ref))==4 % it is an array of autocorrelation matrices already
        covmats_ref = data_ref; 
    else
        covmats_ref = getAllCovMats(data_ref,T_ref,options);
    end
    covmats_ref = sum(covmats_ref,4); 
else
    covmats_ref = [];
end

score = -Inf;
scorepath = cell(options.noruns,1);
if ~isfield(options,'nbatch'), options.nbatch = ndim; end
if ~isfield(options,'maxcyc'), options.maxcyc = 100*N*ndim; end
if options.maxcyc<options.mincyc
    error('maxcyc cannot be lower than mincyc')
end

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
            [scorer,scorers] = getscore(covmats,[],[],covmats_ref);
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
            for j = 1:N
                cm = covmats(:,:,:,j); 
                sm = signmats(:,:,j);
                flipsr(j,d) = ~flipsr(j,d); % do ...
                signmats(:,:,j) = getSignMat(flipsr(j,:)); 
                covmats(:,:,:,j) = applySign(covmats_unflipped(:,:,:,j),signmats(:,:,j));
                s = getscore(covmats,j,scorers,covmats_ref); % ... evaluate ...
                covmats(:,:,:,j) = cm;
                signmats(:,:,j) = sm;
                flipsr(j,d) = ~flipsr(j,d); % ... and undo
                ScoreMatrix(j,d) = s - scorer;
            end
        end
        [score1,I] = max(ScoreMatrix(:));
        [j,d] = ind2sub([N ndim],I);
        if score1>0
            flipsr(j,d) = ~flipsr(j,d);
            if options.verbose
                fprintf('Run %d, Cycle %d, score +%f, flipped (%d,%d) \n',r,cyc,score1,j,d)
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
for j = 1:N
    if mean(flips(j,:))>0.5
        flips(j,:) = 1 - flips(j,:);
    end
end

end


function covmats_unflipped = getAllCovMats(data,T,options)

N = length(T);
if iscell(data)
    for j = 1:N
        if ischar(data{j})
            fsub = data{j};
            loadfile_sub;
        else
            X = data{j};
        end
        Tj = T{j}; Nj = length(Tj);
        if options.standardise
            for jj = 1:Nj
                ind = (1:Tj(jj)) + sum(Tj(1:jj-1));
                X(ind,:) = bsxfun(@minus,X(ind,:),mean(X(ind,:)));
                sd = std(X(ind,:));
                if any(sd==0)
                    error('At least one channel in at least one trial has variance equal to 0')
                end
                X(ind,:) = X(ind,:) ./ repmat(sd,Tj(jj),1);
            end
        end
        covmats_unflipped_j = getCovMats(X,Tj,options.maxlag,options.partial);
        if j==1
            covmats_unflipped = covmats_unflipped_j;
        else % do some kind of weighting here according to the number of samples?
            covmats_unflipped = cat(4,covmats_unflipped,covmats_unflipped_j);
        end
    end
    
else
    if isstruct(data), data = data.X; end
    if options.standardise
        for j = 1:N
            ind = (1:T(j)) + sum(T(1:j-1));
            data(ind,:) = bsxfun(@minus,data(ind,:),mean(data(ind,:)));
            sd = std(data(ind,:));
            if any(sd==0)
                error('At least one channel in at least one trial has variance equal to 0')
            end
            data(ind,:) = data(ind,:) ./ repmat(sd,T(j),1);
        end
    end
    covmats_unflipped = getCovMats(data,T,options.maxlag,options.partial);
end

end












