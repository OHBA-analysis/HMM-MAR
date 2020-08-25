function [A,B,e,e_subj] = highdim_pca(X,T,d,embeddedlags,...
    standardise,onpower,varimax,detrend,filter,leakagecorr,Fs,As)
% pca for potentially loads of subjects
%
% if X is a cell of things, uses SVD
% if X is a matrix, uses Matlab's PCA
%
% d indicates how many PCA components to take, and can be specified
%   in different ways:
% if length(d)==1 and d is lower than 1, then that's the proportion of
%   variance to keep
% if length(d)==1 and d is higher than 1, this is the number of components
% if d is a vector (d1,d2), then d1 must be <1 and d2 must be >=1 ;
%   in this case it will take the minimum of d2 and the number of
%   components that explain d1 amount of variance.
% if d is a vector (d1,d2,d3), is the same than before, but d3 indicates
%   whether to take the minimum (d3=1) or the maximum (d3=2)
%
% Author: Diego Vidaurre, University of Oxford (2016)

if nargin<3, d = []; end
if nargin<4, embeddedlags = 0; end
if nargin<5, standardise = 0; end
if nargin<6, onpower = 0; end
if nargin<7, varimax = 0; end
if nargin<8, detrend = 0; end
if nargin<9, filter = []; end
if nargin<10, leakagecorr = 0; end
if nargin<11, Fs = 1; end
if nargin<12, As = []; end

if length(embeddedlags)>1, msg = '(embedded)';
else, msg = ''; 
end

is_cell_strings = iscell(X) && ischar(X{1});
is_cell_matrices = iscell(X) && ~ischar(X{1});
options = struct();
options.filter = filter; 
options.Fs = Fs; 
options.standardise = standardise;
options.embeddedlags = embeddedlags;
options.pca = 0; % PCA is done here!
options.onpower = onpower;
options.leida = 0;
options.detrend = detrend;
options.leakagecorr = leakagecorr;
options.downsample = 0; % this is always done after PCA

if isfield(options,'A'), options = rmfield(options,'A'); end
if ~isempty(As), options.As = As; end
verbose = 0; 

if is_cell_strings || is_cell_matrices
    B = [];
    for i = 1:length(X)
        X_i = loadfile(X{i},T{i},options); % preproc is done here
        X_i = bsxfun(@minus,X_i,mean(X_i)); % must center
        if isempty(d), d = size(X_i,2); verbose = 0; end
        if i==1, C = zeros(size(X_i,2)); end
        C = C + X_i' * X_i;
    end
    [A,e,~] = svd(C);
    e = diag(e);
else
    X = standardisedata(X,T,options.standardise);
    if ~isempty(options.filter)
        X = filterdata(X,T,options.Fs,options.filter);
    end
    if options.detrend
        X = detrenddata(X,T);
    end
    if options.leakagecorr ~= 0
        X = leakcorr(X,T,options.leakagecorr);
    end
    if options.onpower
        X = rawsignal2power(X,T);
    end
    if options.leida
        X = leadingPhEigenvector(X,T);
    end
    if isfield(options,'As') && ~isempty(options.As)
        X = bsxfun(@minus,X,mean(X)); % must center
        X = X * options.As;
    end
    if length(options.embeddedlags)>1
        X = embeddata(X,T,options.embeddedlags);
    end
    if isstruct(X)
        if isempty(d), d = size(X.X,2); verbose = 0; end
        [A,B,e] = pca(X.X,'Centered',true);
    else
        if isempty(d), d = size(X,2); verbose = 0; end
        [A,B,e] = pca(X,'Centered',true);
    end
end
e = cumsum(e)/sum(e);

if nargout > 3 % explained variance per subject
    e_subj = zeros(length(T),length(e));
    for i = 1:length(T)
        if is_cell_strings || is_cell_matrices
            X_i = loadfile(X{i},T{i},options);
            X_i = bsxfun(@minus,X_i,mean(X_i)); 
        else
            ind = (1:T(i)) + sum(T(1:i-1));
            X_i = X(ind,:);
        end
        Y = zeros(size(X_i));
        for j = 1:length(e)
            Y = Y + X_i * A(:,j) * A(:,j)';
            e_subj(i,j) = sum(sum((X_i - Y).^2));
        end
        var_X_i = sum(sum(X_i.^2));
        e_subj(i,:) = 1 - e_subj(i,:) ./ var_X_i;
        disp(['Subject ' num2str(i)])
    end
end
        
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

if ncomp > size(A,2)
   ncomp = size(A,2);
   warning(['The number of required PCA components is higher than ' ...
       'the dimension of the data - Ignoring PCA.']) 
end

A = A(:,1:ncomp);
if varimax, A = rotatefactors(A); end
if verbose
    if varimax
        fprintf('Working in PCA/Varimax %s space, with %d components. \n',msg,ncomp)
        fprintf('(explained variance = %1f)  \n',e(ncomp))
    else
        fprintf('Working in PCA %s space, with %d components. \n',msg,ncomp)
        fprintf('(explained variance = %1f)  \n',e(ncomp))
    end
end

if ~isempty(B)
    if varimax
        if isstruct(X)
            B = bsxfun(@minus,X.X,mean(X.X)) * A;
        else
            B = bsxfun(@minus,X,mean(X)) * A;
        end
    else
        B = B(:,1:ncomp);
    end
end    

end