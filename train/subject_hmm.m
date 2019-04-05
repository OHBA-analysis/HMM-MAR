function hmm = subject_hmm(data,T,hmm,Gamma,Xi)  
% Get subject-specific states
% If argument Xi is provided, it will also update the transition probability
% matrix
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2016)

options = hmm.train;
checkdatacell
data = data2struct(data,T,options);

% set data
N = length(T);
if iscell(data)
    if size(data,1)==1, data = data'; end
    data = cell2mat(data);
end
if iscell(T)
    if size(T,1)==1, T = T'; end
    T = cell2mat(T);
end
if ~isstruct(data), data = struct('X',data); end

% Filtering
if ~isempty(options.filter)
    data = filterdata(data,T,options.Fs,options.filter);
end
% Detrend data
if options.detrend
    data = detrenddata(data,T);
end
% Standardise data and control for ackward trials
data = standardisedata(data,T,options.standardise);
% Leakage correction
if options.leakagecorr ~= 0
    data = leakcorr(data,T,options.leakagecorr);
end
% Hilbert envelope
if options.onpower
    data = rawsignal2power(data,T);
end
% Leading Phase Eigenvectors
if options.leida
    data = leadingPhEigenvector(data,T);
end
% pre-embedded PCA transform
if length(options.pca_spatial) > 1 || (options.pca_spatial > 0 && options.pca_spatial ~= 1)
    if isfield(options,'As')
        data.X = bsxfun(@minus,data.X,mean(data.X));
        data.X = data.X * options.As;
    else
        [options.As,data.X] = highdim_pca(data.X,T,options.pca_spatial);
        options.pca_spatial = size(options.As,2);
    end
end
% Embedding
if length(options.embeddedlags) > 1
    [data,T] = embeddata(data,T,options.embeddedlags);
end
% PCA transform
if length(options.pca) > 1 || (options.pca > 0 && options.pca ~= 1)
    if isfield(options,'A')
        data.X = bsxfun(@minus,data.X,mean(data.X));
        data.X = data.X * options.A;
    else
        [options.A,data.X] = highdim_pca(data.X,T,options.pca,0,0,0,options.varimax);
        options.pca = size(options.A,2);
    end
    % Standardise principal components and control for ackward trials
    data = standardisedata(data,T,options.standardise_pc);
    options.ndim = size(options.A,2);
    options.S = ones(options.ndim);
    options.Sind = formindexes(options.orders,options.S);
    if ~options.zeromean, options.Sind = [true(1,size(options.Sind,2)); options.Sind]; end
else
    options.ndim = size(data.X,2);
end
% Downsampling
if options.downsample > 0
    [data,T] = downsampledata(data,T,options.downsample,options.Fs);
end
% get global eigendecomposition
if options.firsteigv
    if isstruct(data)
        data.X = bsxfun(@minus,data.X,mean(data.X));
        options.gram = data.X' * data.X;
    else
        data = bsxfun(@minus,data,mean(data));
        options.gram = X' * X;
    end
    [options.eigvec,options.eigval] = svd(options.gram);
    options.eigval = diag(options.eigval);
end
if options.pcamar > 0 && ~isfield(options,'B')
    % PCA on the predictors of the MAR regression, per lag: X_t = \sum_i X_t-i * B_i * W_i + e
    options.B = pcamar_decomp(data,T,options);
end
if options.pcapred > 0 && ~isfield(options,'V')
    % PCA on the predictors of the MAR regression, together:
    % Y = X * V * W + e, where X contains all the lagged predictors
    % So, unlike B, V draws from the temporal dimension and not only spatial
    options.V = pcapred_decomp(data,T,options);
end

setxx;
if ~isfield(hmm.train,'Sind')
    orders = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
    hmm.train.Sind = formindexes(orders,hmm.train.S);
end
residuals =  getresiduals(data.X,T,hmm.train.Sind,hmm.train.maxorder,hmm.train.order,...
    hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag,hmm.train.zeromean);

% obtain observation model
hmm = obsupdate (T,Gamma,hmm,residuals,XX,XXGXX,1);

% obtain transition probabilities
if nargin==5
    hmm = hsupdate(Xi,Gamma,T,hmm);
end

end
