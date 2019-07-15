function [C,A,e] = corrStaticFC (data,T,options,use_pca)
% When running the HMM on fMRI data (with order=0, i.e. a Gaussian
% distribution with mean and covariance), 
% it can happen that states are assigned to entire subjects
% with not much switching between states. This happens typically when the
% static functional connectivity (FC) is so different that states specialise 
% into specific subjects, explaining these grand patterns with
% no room to capture any dynamic FC. 
% This function computes the (subjects by subjects) matrix of static FC 
% similarities (measured in terms of correlation) between each pair of
% subjects. If the obtained values are too low, then covtype='uniquefull'
% has a higher chance to do a good job. 
%
% If options.embeddedlags is specified, then it compares the time-embedded
% covariance matrices in a similar fashion; see Vidaurre et al. (2018) Nature
% Communications. 
% 
% If the use_pca parameter is set to 1, then C will be 
% (subjects by subjects by no. of PCA components), with one matrix of
% similarities per number of PCA components; e.g. C(:,:,3) is the
% similarity matrix when using three PC; C(:,:,4) is the
% similarity matrix when using four PCs, etc. Finally, C(:,:,end) is the
% similarity marix in the original space (no PCA). 
% C(:,:,1) and C(:,:,2) are set to NaN.
% In that case, A is the PCA mixing matrix, and e the explained variance
% for each number of components; e.g. e(3) is the explained variance for 3
% PCA components.
% 
% Author: Diego Vidaurre (2017)

if nargin<3, options = struct(); end
if nargin<4, use_pca = 0; end
%if nargin<5, toPlot = 1; end

if xor(iscell(data),iscell(T)), error('X and T must be cells, either both or none of them.'); end

if iscell(T)
    if size(T,1)==1, T = T'; end
    for i = 1:length(T)
        if size(T{i},1)==1, T{i} = T{i}'; end
        T{i} = int64(T{i});
    end
    if size(data,1)==1, data = data'; end
else
    T = int64(T);
end
N = length(T);

options.K = 2; options.order = 0; 
options.BIGNbatch = 2; % to avoid checkoptions complaining
options = checkspelling(options);
options = checkoptions(options,data,T,0);

if length(options.pca) > 1 || (options.pca > 0 && options.pca ~= 1)
    warning('options.pca will be ignored; use use_pca instead')
end

if use_pca
    [A,~,e] = highdim_pca(data,T,[],...
        options.embeddedlags,options.standardise,...
        options.onpower,options.varimax,options.detrend,...
        options.filter,options.leakagecorr,options.Fs);
    npca = size(A,2); 
    if isfield(options,'A'), warning('options.A will be ignored; use use_pca instead'); end
else
    A = []; e = []; 
end

if use_pca
    FC = cell(npca+1,1); % last one is without PCA
end

for j = 1:N
    if iscell(data)
        if ischar(data{j})
            dat = load(data{j});
            if isfield(dat,'X')
                X = dat.X;
            else
                X = getfield(dat,char(fieldnames(dat)));
            end
        else
            X = data{j};
        end
        Tj = T{j};
    else
        ind = (1:T(j)) + sum(T(1:j-1));
        if isstruct(data)
            X = data.X(ind,:);
        else
            X = data(ind,:);
        end
        Tj = T(j);
    end
    % Standardise data and control for ackward trials
    X = standardisedata(X,Tj,options.standardise);  
    % Filtering
    if ~isempty(options.filter)
       X = filterdata(X,Tj,options.Fs,options.filter);
    end
    % Detrend data
    if options.detrend
       X = detrenddata(X,Tj); 
    end   
    % Leakage correction
    if options.leakagecorr ~= 0 
        X = leakcorr(X,Tj,options.leakagecorr);
    end
    % Hilbert envelope
    if options.onpower
       X = rawsignal2power(X,Tj); 
    end
    % Embedding
    if length(options.embeddedlags) > 1  
        [X,Tj] = embeddata(X,Tj,options.embeddedlags);
    end
    if use_pca
        X_nopca = X;
        ndim = size(X,2); 
        if j==1, FC =  zeros(ndim*(ndim-1)/2,N,npca+1); end
        for d = 1:npca
            X = X_nopca * A(:,1:d) * A(:,1:d)';
            X = X + 1e-5 * randn(size(X)); % add some noise to avoid ill-conditioning
            % Downsampling
            if options.downsample > 0
                X = downsampledata(X,Tj,options.downsample,options.Fs);
            end
            c = corr(X);
            FC(:,j,d) =  c(triu(true(ndim),1))';
        end
        X = X_nopca; 
        if options.downsample > 0
            X = downsampledata(X,Tj,options.downsample,options.Fs);
        end
        c = corr(X);
        FC(:,j,end) =  c(triu(true(ndim),1))';
    else
        % PCA transform (precomputed)
        if isfield(options,'A') && ~isempty(options.A)
            X = X * A;
        end
        % Downsampling
        if options.downsample > 0
            X = downsampledata(X,Tj,options.downsample,options.Fs);
        end
        c = corr(X);
        if j==1
            ndim = size(X,2);
            FC = zeros(ndim*(ndim-1)/2,N);
        end
        FC(:,j) =  c(triu(true(ndim),1))';
    end
end

if use_pca
    C = NaN(N,N,size(FC,3));
    for d = 1:size(FC,3)
        c = corr(FC(:,:,d));
        c(eye(N)==1) = Inf;
        C(:,:,d) = c; 
    end
else
    C = corr(FC);
    C(eye(N)==1) = Inf;
end

% if toPlot
%     imagesc(C,[-1 1]); colorbar
%     set(gca,'FontSize',18)
%     colormap('jet');
%     grotc=colormap;  grotc(end,:)=[.8 .8 .8];  colormap(grotc);
% end

end
