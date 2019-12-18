function [flips,scorepath,covmats_unflipped] = findflip(data,T,options)
% Finds an optimal permutation of the channels, where goodness is measured
% as the mean lagged partial  cross-correlation across pair of channels and lags.
% In other words, it finds a permutation where the signs of the lagged
% partial correlations agree as much as possible across subjects.
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
%  nbatch        no. of channels to evaluate at each iteration (0 for all)
%  noruns        how many random initialisations will be carried out
%  standardise   if 1, standardise the data
%  partial       if 1, base on partial correlation instead of correlation
%  maxcyc        for each initialization, maximum number of cycles of the greedy algorithm
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

if nargin < 3, options = struct; end

options = checkoptions_flip(options);
covmats_unflipped = Get_Global_Variables_For_BitFlip_Evaluation(data,T,options);
N = size(covmats_unflipped,4);

ndim = size(covmats_unflipped,1);
score = -Inf;
scorepath = cell(options.noruns,1);

if isinf(options.maxcyc) && (options.nbatch > 0)
    error('If maxcyc is Inf, options.nbatch must be 0')
end

% repetitions of the search
for r = 1:options.noruns
    
    flipsr = Init_Solution(N,ndim,options,r);
    score_r = EvaluateFlips(flipsr,covmats_unflipped);
    scorepath{r} = score_r;
    if options.verbose
        fprintf('Run %d, Init, score %f \n',r,score_r)
    end
        
    for cyc = 1:options.maxcyc
      
        if options.nbatch > 0
            channels = randperm(ndim,options.nbatch);
        else
            channels = randperm(ndim);
        end
             
        ScoreMatrix = zeros(N,ndim);
        for d = channels
            for j = 1:N
                ScoreMatrix(j,d) = EvaluateFlips(flipsr,covmats_unflipped,j,d);
            end
        end
        
        [score_r,I] = max(ScoreMatrix(:));
        [j,d] = ind2sub([N ndim],I);
        
        ds = score_r - max(scorepath{r});
        if ds > options.threshold
            flipsr(j,d) = ~flipsr(j,d);
            if options.verbose
                fprintf('Run %d, Cycle %d, score +%f, flipped (%d,%d) \n',r,cyc,ds,j,d)
            end
            scorepath{r} = [scorepath{r} score_r];
        elseif options.nbatch == 0
            break
        else
            fprintf('Run %d, Cycle %d, score +0 \n',r,cyc)
        end
        
    end
    
    if options.verbose
        fprintf('Run %d, Finish, score %f \n',r,scorepath{r}(end))
    end
    
    if scorepath{r}(end) > score
        score = scorepath{r}(end);
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


function options = checkoptions_flip(options)
if ~isfield(options,'maxlag'), options.maxlag = 10; end
if ~isfield(options,'noruns'), options.noruns = 1; end
if ~isfield(options,'probinitflip'), options.probinitflip = 0.25; end
if ~isfield(options,'standardise'), options.standardise = 1; end
if ~isfield(options,'partial'), options.partial = 0; end
if ~isfield(options,'verbose'), options.verbose = 1; end
if ~isfield(options,'nbatch'), options.nbatch = 0; end
if ~isfield(options,'maxcyc'), options.maxcyc = 10000; end
if ~isfield(options,'threshold'), options.threshold = 0.00001; end
end


function flipsr = Init_Solution(N,ndim,options,r)
% Prepare the necessary data structures
if ~isfield(options,'Flips')
    if r==1 % first run starts at no flips
        flipsr = zeros(N,ndim);
    else
        flipsr = binornd(1,options.probinitflip,N,ndim); % random init
    end
else
    flipsr = options.Flips;
end
end


function covmats_unflipped = Get_Global_Variables_For_BitFlip_Evaluation(data,T,options)
% Prepare the necessary data structures
if length(size(data))==4 % it is an array of autocorrelation matrices already
    covmats_unflipped = data; clear data
else
    covmats_unflipped = getAllCovMats(data,T,options);
end
end


function s = EvaluateFlips(flipsr,covmats_unflipped,j,d)
% evaluate for a change in subject j and channel d
if nargin>3, flipsr(j,d) = ~flipsr(j,d);  end
signmats = getSignMat(flipsr);
covmats = applySign(covmats_unflipped,signmats);
s = getscore(covmats);
end


function score = getscore(M)
% get the score from the (flipped) matrices of autocorrelation contained in M
% M is a matrix (ndim x ndim x lags x subjects) with autocovariances matrix for all subjects
N = size(M,4); ndim = size(M,1); L = size(M,3);
M = reshape(M,[(ndim^2)*L N]);
C = corr(M);
score = mean(C(triu(true(N),1)));
end


function CovMats = applySign(CovMats,SignMats)
% apply matrices of sign flipping to the autocorrelation matrices
N = size(SignMats,3); nlags = size(CovMats,3);
for j = 1:N
    CovMats(:,:,:,j) = CovMats(:,:,:,j) .* repmat(SignMats(:,:,j),[1 1 nlags]);
end
end


function SignMats = getSignMat(Flips)
% Construct matrices of sign flipping for the autocorrelation matrices
[N,ndim] = size(Flips);
SignMats = zeros(ndim,ndim,N);
for j = 1:N 
    flips = ones(1,ndim);
    flips(Flips(j,:)==1) = -1;
    SignMats(:,:,j) = flips' * flips;
end
end
