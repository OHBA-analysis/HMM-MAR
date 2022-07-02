function [flips,scorepath,covmats_unflipped,flips_init] = findflip_bayes(data,T,options)
% 
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
% Author: Cam Higgins and Diego Vidaurre, University of Oxford.

if nargin < 3, options = struct; end

options = checkoptions_flip(options);
%covmats_unflipped = Get_Global_Variables_For_BitFlip_Evaluation(data,T,options);
covmats_unflipped = get_big_covmats(data,T,options);
%covmats_unflipped_diego = Get_Global_Variables_For_BitFlip_Evaluation(data,T,options);
[ndim,~,N] = size(covmats_unflipped);
nch = ndim./(2*options.maxlag+1);

nembed = ndim/nch;
scorepath = cell(options.noruns,1);

mincyc = 5;

% repetitions of the search (if necessary):
for r = 1:options.noruns

    % initialise model object:
    SFmodel = [];
    SFmodel.V0 = eye(ndim); % template matrix prior scale matrix
    SFmodel.chi = ndim; % template matrix degrees of freedom
    SFmodel.P0 = 0.5; % prior probability of flips
    SFmodel.nch = nch;
    
    if ~options.templateinit 
        %initialise posterior flip probabilities:
        Pinit = 1-0.5*rand(N,SFmodel.nch);
        SFmodel.P = Pinit; % initialise conservatively so no dimensions are flipped at initialisation (but some are downweighted)
        %SFmodel.P = ones(N,SFmodel.nch); % initialise so nothing is flipped

        % infer first template matrix:
        SFmodel.S = updateS(SFmodel,covmats_unflipped);
    else
        % alternative init to template subject:
        offdiagblocks = triu(ones(ndim) - repelem(eye(nch),nembed,nembed));
        for iSj=1:N
            subjmetrics(iSj) = sum(sum(abs(covmats_unflipped(:,:,iSj).*offdiagblocks)));
            distmet(:,iSj) = squash(abs(covmats_unflipped(:,:,iSj).*offdiagblocks));
        end
        [~,templatesub] = max(subjmetrics);
        D = dist(distmet);
        Pinit = zeros(N,nch);
        Pinit(templatesub,:) = 1;
        SFmodel.P = Pinit(templatesub,:); % do not flip template subject
        SFmodel.S = updateS(SFmodel,covmats_unflipped(:,:,templatesub));
        for iSj=setdiff(1:N,templatesub)
            SFmodel.P = ones(1,nch);
            for i=1:5
                Psub = updateP(SFmodel,covmats_unflipped(:,:,iSj));
                SFmodel.P = Psub;
            end
            Pinit(iSj,:) = Psub;       
        end
        SFmodel.P = Pinit;
    end
    
    flips_init = Pinit<0.5;
    exp_invS = SFmodel.S.chi * SFmodel.S.V;
    for subnum=1:N
        D = repelem(1-2*flips_init(subnum,:)',nembed,1);
        LL(subnum) = -0.5*trace(exp_invS*[(D*D').*covmats_unflipped(:,:,subnum)]);
    end
    LLtot(1,1) = sum(LL);
    
    repcount = 0;
    for cyc = 1:options.maxcyc
      
        % if batching over subjects, do here
        if options.nbatch > 0
             sjs_thisbatch = sort(randperm(N,options.nbatch));
        else
             sjs_thisbatch = 1:N;
        end
        SFmodel.P = updateP(SFmodel,covmats_unflipped,sjs_thisbatch);
        SFmodel.S = updateS(SFmodel,covmats_unflipped);
        
        flips_new = SFmodel.P<0.5;
        flips_debugrec(:,:,cyc) = flips_new;

        if cyc>mincyc && all(flips_new(:)==flips_bayes(:))
            repcount = repcount+1;
            if repcount>0
                break;
            end
        else
            flips_bayes = flips_new;
            repcount = 0;
        end
        
        % eval expected log likelihood:
        exp_invS = SFmodel.S.chi * SFmodel.S.V;
        for subnum=1:N
            D = repelem(1-2*flips_bayes(subnum,:)',nembed,1);
            LL(subnum) = -0.5*trace(exp_invS*[(D*D').*covmats_unflipped(:,:,subnum)]);
        end
        LLtot(cyc+1,1) = sum(LL);

    end

    flips = flips_bayes;
end


% Among the equivalent flippings, we keep the one w/ the lowest no. of flips
for j = 1:nch
    if mean(flips(:,j))>0.5
        flips(:,j) = 1 - flips(:,j);
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

function covmats_unflipped = get_big_covmats(data,T,options)
if ~isfield(options,'maxlag'), options.maxlag = 10; end
if ~isfield(options,'partial'), options.partial = 0; end
if ~isfield(options,'standardise'), options.standardise = 1; end

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
        covmats_unflipped_j = getCovMats_big(X,sum(Tj),options.maxlag);
        if j==1
            covmats_unflipped = covmats_unflipped_j;
        else % do some kind of weighting here according to the number of samples?
            covmats_unflipped = cat(3,covmats_unflipped,covmats_unflipped_j);
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
    covmats_unflipped = getCovMats_big(data,T,options.maxlag);
end
end

function CovMats = getCovMats_big(X,T,maxlag)
N = length(T); ndim = size(X,2);
if nargin<5, Flips = zeros(N,ndim); end
CovMats = zeros(ndim*(2*maxlag+1),ndim*(2*maxlag+1),N);
t0 = 0;
for j = 1:N 
    t1 = t0 + T(j);
    Xj = X(t0+1:t1,:); t0 = t1;
    for i = 1:ndim, if Flips(j,i)==1, Xj(:,i) = -Xj(:,i); end; end
    Y = embeddata(Xj,size(Xj,1),-maxlag:maxlag);
    CovMats(:,:,j) = cov(Y);
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

function S = updateS(SFmodel,covmats_unflipped)
% update factorised posterior estimate for global template covmat:
S = [];
S.V = inv(SFmodel.V0); % prior
[ndim,~,nSj] = size(covmats_unflipped);
nch = size(SFmodel.P,2);
nembed = ndim/nch;
for i=1:nSj
    Pmat = (2*SFmodel.P(i,:)-1)'*(2*SFmodel.P(i,:)-1);
    weightmat = kron(Pmat.*(ones(nch)-eye(nch)),ones(nembed)) + kron(eye(nch),ones(nembed));
    S.V = S.V + weightmat.*covmats_unflipped(:,:,i);
end
S.chi = (nSj+1)*ndim;
%S.chi = nch;
end

function [P,L] = updateP(SFmodel,covmats_unflipped,sjs_this_batch)
% update factorised posterior estimate for sign flip probability:
if nargin<3
    sjs_this_batch = 1:size(covmats_unflipped,3);
end
P = SFmodel.P;
[ndim,~,nSj] = size(covmats_unflipped);
nch = size(SFmodel.P,2);
nembed = ndim/nch;
exp_invS = SFmodel.S.chi * SFmodel.S.V;
for iSj=sjs_this_batch
    fullmat = exp_invS.*covmats_unflipped(:,:,iSj);
    chanorder = randperm(nch);
    for ich = chanorder
        chinds = (ich-1)*nembed + [1:nembed];
        chouts = setdiff(1:ndim,chinds);
        %sum(sum(fullmat(chinds,chouts).*kron(P(iSj,setdiff(1:nch,ich)),ones(nembed))))
        L(iSj,ich) = sum(reshape(fullmat(chinds,chouts),[nembed*nembed,nch-1])*((2*P(iSj,setdiff(1:nch,ich))')-1));
        % debugging option: binarise P in estimates:
        %L(iSj,ich) = sum(reshape(fullmat(chinds,chouts),[nembed*nembed,nch-1])*((2*P(iSj,setdiff(1:nch,ich))')-1)>0);
        % compute probability:
        P(iSj,ich) = 1 ./ (1+exp(-2*L(iSj,ich)));
    end
end

end

% function chi = updatechi(SFmodel,covmats_unflipped)
% % implement MAP point estimate update for the degrees of freedom chi
% lambda0 = 1;
% [ndim,~,nSj] = size(covmats_unflipped);
% nch = size(SFmodel.P,2);
% nembed = ndim/nch;
% covsum = 0;
% for iSj=1:nSj
%     covsum = covsum + logdet(covmats_unflipped(:,:,iSj));
% end
% derivfunc = @(chi) nSj*(multivariatepsi(0.5*SFmodel.S.chi,ndim)-psi(chi)+logdet(SFmodel.S.V)) - 2*lambda0+covsum;
% 
% chi = fzero(derivfunc,lambda0);
% end
% 
% function multipsi = multivariatepsi(a,ndim)
% % implements the multivariate diigamma function: 
% % https://www.wikiwand.com/en/Multivariate_gamma_function#/Derivatives
% if a<(ndim-1)
%     error('Not full rank!');
% end
% multipsi = 0;
% for i=1:ndim
%     multipsi = multipsi + psi(a+0.5*(1-i));
% end
% end