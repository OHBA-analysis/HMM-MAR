function [X,Y,T,options,R2_pca,pca_opt,features] = preproc4hmm(X,Y,T,options)
% Prepare data to run TUDA

if length(size(X))==3 % 1st dim, time; 2nd dim, trials; 3rd dim, channels
    X = reshape(X,[size(X,1)*size(X,2), size(X,3)]);
end

q = size(Y,2);
N = length(T);
p = size(X,2);

if size(X,1) ~= sum(T), error('Dimension of X not correct'); end
if (size(Y,1) ~= sum(T)) && (size(Y,1) ~= length(T))
    error('Dimension of Y not correct');
end

if isfield(options,'downsample') && options.downsample
   error('Downsampling is not currently an option') 
end
if isfield(options,'filter') && ~isempty(options.filter) && ~isfield(options,'Fs')
   error('You must specify options.Fs if you are to filter the data') 
end

% options relative to the regression setting
if ~isfield(options,'Nfeatures'), Nfeatures = p;
else, Nfeatures = options.Nfeatures; end
if ~isfield(options,'standardise'), standardise = 1;
else, standardise = options.standardise; end
if ~isfield(options,'onpower'), onpower = 0;
else, onpower = options.onpower; end
if ~isfield(options,'embeddedlags'), embeddedlags = 0;
else, embeddedlags = options.embeddedlags; end
if ~isfield(options,'filter'), filter = [];
else, filter = options.filter; end
if ~isfield(options,'detrend'), detrend = 0;
else, detrend = options.detrend; end
% econ_embed saves memory at the expense of speed, when pca is applied
if ~isfield(options,'econ_embed'), econ_embed = 0;
else, econ_embed = options.econ_embed; end
if ~isfield(options,'parallel_trials'), parallel_trials = all(T==T(1));
else, parallel_trials = options.parallel_trials; end
if ~isfield(options,'pca'), pca_opt = 0;
else, pca_opt = options.pca; end
if ~isfield(options,'add_noise'), add_noise = 1;
else, add_noise = options.add_noise; end

% Options relative to constraints in the trans prob mat
if ~isfield(options,'K'), error('K was not specified'); end
if ~isfield(options,'Pstructure')
    options.Pstructure = true(options.K);
end
if ~isfield(options,'Pistructure')
    options.Pistructure = true(1,options.K);
end

options.parallel_trials = parallel_trials;
if ~isfield(options,'tudamonitoring'), options.tudamonitoring = 0; end

if parallel_trials && ~all(T==T(1))
    error('parallel_trials can be used only when all trials have equal length');
end
if options.tudamonitoring && ~all(T==T(1))
    error('tudamonitoring can be used only when all trials have equal length');
end

% options relative to the HMM
if ~isfield(options,'covtype'), options.covtype = 'uniquediag'; end
options.order = 1;
options.zeromean = 1;
options.embeddedlags = 0; % it is done here
options.pca = 0; % it is done here
options.standardise = 0; % it is done here
options.onpower = 0; % it is done here
options.detrend = 0; % it is done here
options.filter = []; 
options.downsample = 0; 
options.dropstates = 0;

options.inittype = 'HMM-MAR';
if isfield(options,'econ_embed'), options = rmfield(options,'econ_embed'); end
if isfield(options,'Nfeatures'), options = rmfield(options,'Nfeatures'); end

do_embedding = length(embeddedlags)>1;
do_pca = length(pca_opt)>1 || (pca_opt>0 && pca_opt<p);

if ~do_embedding && econ_embed
    econ_embed = 0;
end
if do_embedding && ~do_pca && econ_embed
    warning('It only makes sense to use econ_embed when using pca')
    econ_embed = 0;
end

emforw = max(max(embeddedlags),0);
emback = max(-min(embeddedlags),0);

if size(Y,1) == N % one value for the entire trial
    Ytmp = Y;
    Y = zeros(sum(T),q);
    for n = 1:N
        Y(sum(T(1:n-1)) + (1:T(n)),:) = repmat(Ytmp(n,:),T(n),1);
    end; clear Ytmp
    Y = Y + 1e-6 * repmat(std(Y),size(Y,1),1) .* randn(size(Y)) ;
end


% Demean stimulus
Y = bsxfun(@minus,Y,mean(Y));
if add_noise % this avoids numerical problems 
   Y = Y + 1e-4 * randn(size(Y)); 
end
% Filtering
if ~isempty(filter)
    data = filterdata(X,T,options.Fs,filter);
end
% Detrend data
if detrend
    X = detrenddata(X,T);
end
% Standardise data
X = standardisedata(X,T,standardise);

% adjust dimension of Y according to embedding
if do_embedding
    Ttmp = T-emforw-emback;
    Ytmp = Y;
    Y = zeros(sum(Ttmp),q);
    for n = 1:N
        Y(sum(Ttmp(1:n-1)) + (1:Ttmp(n)),:) = ...
            Ytmp(sum(T(1:n-1)) + (1+emforw:T(n)-emback) ,:);
    end; clear Ytmp Ttmp
end

% feature selection
if Nfeatures < p && Nfeatures > 0
    me = sum((repmat(mean(Y),size(Y,1),1) - Y).^2);
    C = zeros(p,1);
    for j=1:p
        if onpower
            x = rawsignal2power(X(:,j),T);
        else
            x = X(:,j);
        end
        if do_embedding
            x = embeddata(x,T,embeddedlags);
        end
        b = (x' * x) \ (x' * Y);
        e = sum((x * b - Y).^2);
        C(j) = sum(1 - e ./ me);
    end
    [~,features] = sort(C,1,'descend');
    features = features(1:Nfeatures);
    X = X(:,features);
    p = Nfeatures;
else
    features = 1:p;
end

% Hilbert envelope
if onpower
    X = rawsignal2power(X,T);
end

% do embedding + PCA
R2_pca = []; 
if econ_embed
    
    % build gram matrix subject by subject
    for n = 1:N
        t = (1:T(n)) + sum(T(1:n-1));
        if do_embedding
            Xn = embeddata(X(t,:),T(n),embeddedlags);
        else
            Xn = X(t,:);
        end
        Xn = Xn - repmat(mean(Xn),size(Xn,1),1); % must center
        if n==1, C = zeros(size(Xn,2)); end
        C = C + Xn' * Xn;
    end
    
    % do SVD
    [A,e,~] = svd(C);
    e = diag(e);
    e = cumsum(e)/sum(e);
    p = num_comp_pca(e,pca_opt);
    A = A(:,1:p);
    R2_pca = e(p);
    
    % eigendecompose subject by subject
    Xtmp = X; Ttmp = T;
    T = T-emforw-emback;
    X = zeros(sum(T),p);
    for n = 1:N
        t = (1:T(n)) + sum(T(1:n-1));
        ttmp = (1:Ttmp(n)) + sum(Ttmp(1:n-1));
        if do_embedding
            Xn = embeddata(Xtmp(ttmp,:),Ttmp(n),embeddedlags);
        else
            Xn = Xtmp(ttmp,:);
        end
        Xn = Xn - repmat(mean(Xn),size(Xn,1),1); % must center
        X(t,:) = Xn * A;
    end
    
else
    
    if do_embedding
        [X,T] = embeddata(X,T,embeddedlags);
        msg = '(embedded)';
    else
        msg = '';
    end
    if do_pca
        [~,X,e] = pca(X);
        e = cumsum(e)/sum(e);
        p = num_comp_pca(e,pca_opt);
        R2_pca = e(p);
        X = X(:,1:p);
        fprintf('Working in PCA %s space, with %d components. \n',msg,p)
    else
        R2_pca = 1;
    end
    
end

end


function ncomp = num_comp_pca(e,d)

if length(d)==1 && d<1
    ncomp = find(e>d,1);
elseif length(d)==1 && d>=1
    ncomp = d;
elseif length(d)==2 || (length(d)==3 && d(3)==1)
    ncomp = min(find(e>d(1),1),d(2));
elseif length(d)==3 && d(3)==2
    ncomp = max(find(e>d(1),1),d(2));
else
    error('pca parameters are wrongly specified')
end

end
