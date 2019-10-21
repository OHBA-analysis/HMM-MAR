function [X,Y,T,options,A,R2_pca,pca_opt,features] = preproc4hmm(X,Y,T,options)
% Prepare data to run TUDA

if length(size(X))==3 % 1st dim, time; 2nd dim, trials; 3rd dim, channels
    X = reshape(X,[size(X,1)*size(X,2), size(X,3)]);
end

N = length(T);
p = size(X,2);

if any(isnan(Y(:))), error('NaN found in Y'); end

if size(X,1) ~= sum(T), error('Dimension of X not correct'); end
if size(Y,1) < size(Y,2); Y = Y'; end
if (size(Y,1) ~= sum(T)) && (size(Y,1) ~= length(T))
    error('Dimension of Y not correct');
end
q = size(Y,2);

if isfield(options,'downsample') && options.downsample
   error('Downsampling is not currently an option') 
end
if isfield(options,'filter') && ~isempty(options.filter) && ~isfield(options,'Fs')
   error('You must specify options.Fs if you are to filter the data') 
end

% options relative to the regression setting
if ~isfield(options,'Nfeatures'), Nfeatures = p;
else, Nfeatures = options.Nfeatures; end
if ~isfield(options,'standardise'), standardise = 0;
else, standardise = options.standardise; end
if ~isfield(options,'demeanstim'), demeanstim = 1;
else, demeanstim = options.demeanstim; end
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
if ~isfield(options,'parallel_trials'), parallel_trials = all(T==T(1)) & length(T)>1;
else, parallel_trials = options.parallel_trials; end
if ~isfield(options,'pca'), pca_opt = 0;
else, pca_opt = options.pca; end
if ~isfield(options,'A'), A = [];
else, A = options.A; end
if isfield(options,'downsample') && options.downsample~=0
    warning('Downsampling is not possible for TUDA')
end

%options relative to classification models:
if isfield(options,'classifier')
    if strcmp(options.classifier,'logistic')
        %set default options for logistic regression classification:
        options.distribution='logistic';
        demeanstim=false;
        %determine if multinomial or binomial:
        vals = unique(Y(:));
        if length(vals)==2
            if all(vals == [0;1])
                Y=2*(Y)-1;
            elseif ~all(vals == [-1;1])
                ME = MException(preproc4hmm:wrongYformat,'Error: format of Y incorrect for classification tasks');
                throw ME;
            end
        elseif length(vals)>2 && size(Y,2)==1
            % Y entered as categorical format
            fprintf(['\nFitting multinomial logistic classifier with classes ',int2str(vals'), '\n']);
            Y=convertToMultinomial(Y);
        elseif length(vals)==3
            if ~all(vals == [-1;0;1])
                throw exception;
            end
        end
        options.logisticYdim=size(Y,2);
        if ~isfield(options,'balancedata')
            options.balancedata=0;
        else
            options.balancedata=options.balancedata;
        end
        if ~isfield(options,'intercept'),options.intercept=0;end
        if options.intercept
           X = [X,ones(size(X,1),1)];
        end
        options=rmfield(options,'intercept');
        if ~isfield(options,'sequential')
            options.sequential=true;
        end
        options.add_noise=0;
   elseif strcmp(options.classifier,'LDA')
       % set default options for LDA model:
       options.distribution='Gaussian';
       demeanstim=false;
        if ~isfield(options,'intercept'),options.intercept=1;end
        if options.intercept
            if size(Y,2)==1
                Y=[ones(size(Y,1),1),Y==1,Y~=1];
            else
                Y=[ones(size(Y,1),1),Y];
            end
            q = size(Y,2);
        end
        if ~isfield(options,'covtype')
            options.covtype = 'uniquediag';
        end
        options=rmfield(options,'intercept');
        if ~isfield(options,'sequential')
            options.sequential=true;
        end
        options.add_noise=0;
    elseif strcmp(options.classifier,'SVM')
        options.add_noise=0;
        demeanstim=false;
        options.sequential=false;
    end
    % general classification options:
    if options.sequential
        options.Pstructure = eye(options.K) + diag(ones(1,options.K-1),1);
        options.Pistructure = zeros(1,options.K);
        options.Pistructure(1)=1;
    end
    options = rmfield(options,'classifier');
else
    options.distribution='Gaussian'; %default for all non-classification models
end
if ~isfield(options,'logisticYdim'), options.logisticYdim=0;end
if ~isfield(options,'add_noise'), add_noise = 1;else add_noise = options.add_noise; end


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
if ~isfield(options,'plotAverageGamma'), options.plotAverageGamma = 0; end

if parallel_trials && ~all(T==T(1))
    error('parallel_trials can be used only when all trials have equal length');
end
if options.tudamonitoring && ~all(T==T(1))
    error('tudamonitoring can be used only when all trials have equal length');
end

% options relative to the HMM
if ~isfield(options,'distribution'),options.distribution='Gaussian';end
if options.logisticYdim>0; 
    options.distribution = 'logistic';
end
if strcmp(options.distribution,'logistic')
    options.covtype='';
end
if ~isfield(options,'covtype') && strcmp(options.distribution,'Gaussian')
    options.covtype = 'uniquediag'; 
end

options.order = 1;
options.zeromean = 1;
options.embeddedlags = 0; % it is done here
options.pca = 0; % it is done here
options.standardise = 0; % it is done here
options.onpower = 0; % it is done here
options.detrend = 0; % it is done here
options.filter = []; % it is done here
options.downsample = 0; % it is done here 
options.dropstates = 0;

options.inittype = 'HMM-MAR';
if isfield(options,'econ_embed'), options = rmfield(options,'econ_embed'); end
if isfield(options,'Nfeatures'), options = rmfield(options,'Nfeatures'); end
if isfield(options,'demeanstim'), options = rmfield(options,'demeanstim'); end
% Set a high prior for the initial probabilities because otherwise the
% model is biased to have the first time point of each trial to be assigned
% to just one state.
if ~isfield(options,'PriorWeightingPi'), options.PriorWeightingPi = length(T); end
if ~isfield(options,'DirichletDiag'), options.DirichletDiag = 100; end

do_embedding = length(embeddedlags)>1;
do_pca = ~isempty(A) || length(pca_opt)>1 || (pca_opt>0 && pca_opt<(p*length(embeddedlags)));

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
    %Y = Y + 1e-6 * repmat(std(Y),size(Y,1),1) .* randn(size(Y)) ;
end

if q == 1 && length(unique(Y))==2
    if ismember(0,unique(Y)) || all(unique(Y)>0)
        warning('Seems this is binary classification, transforming stimulus to have elements (-1,+1)')
        v = unique(Y);
        Y(Y==v(1)) = -1; Y(Y==v(2)) = +1;
        if options.logisticYdim==0
            Y = Y - mean(Y);
        end
    end
end

if demeanstim
    % Demean stimulus
    Y = bsxfun(@minus,Y,mean(Y));
    if add_noise % this avoids numerical problems 
       Y = Y + 1e-4 * randn(size(Y)); 
    end
end
% Add noise, to avoid numerical problems 
if add_noise > 0
    if add_noise == 1
        Y = Y + 1e-5 * randn(size(Y)) .* repmat(std(Y),size(Y,1),1);
    else
        Y = Y + add_noise * randn(size(Y)) .* repmat(std(Y),size(Y,1),1);
    end
end
% Standardise data
if standardise && N > 1
   warning(['You have set standardise=1, so each channel and trial will be standardized. ' ...
       'This will probably result in a loss of information in terms of how each stimulus is processed'])
   X = standardisedata(X,T,standardise); 
end

% Filtering
if ~isempty(filter)
    data = filterdata(X,T,options.Fs,filter);
end
% Detrend data
if detrend
    X = detrenddata(X,T);
end

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
    if isempty(A)
        [A,e,~] = svd(C);
        e = diag(e);
        e = cumsum(e)/sum(e);
        p = num_comp_pca(e,pca_opt);
        A = A(:,1:p);
        R2_pca = e(p);
    else
        R2_pca = []; p = size(A,2);
    end
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
        if isempty(A)
            [A,X,e] = pca(X);
            e = cumsum(e)/sum(e);
            p = num_comp_pca(e,pca_opt);
            R2_pca = e(p);
            X = X(:,1:p);
            A = A(:,1:p);
            fprintf('Working in PCA %s space, with %d components. \n',msg,p)
        else
            X = bsxfun(@minus,X,mean(X));   
            X = X * A; 
        end
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
