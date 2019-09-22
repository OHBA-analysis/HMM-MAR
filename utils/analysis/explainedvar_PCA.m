function [e_group,e_subj] = explainedvar_PCA(data,T,options)
% Performs preprocessing on the data, performs PCA,
% and returns the explained variance for each PCA component. 
% This is useful to inspect the rank of the data and decide the number of
% PC components to use in an informed way
% 
% Author: Diego Vidaurre, OHBA, University of Oxford (2015)

if iscell(T)
    if size(T,1)==1, T = T'; end
    for i = 1:length(T)
        if size(T{i},1)==1, T{i} = T{i}'; end
    end
    N = numel(cell2mat(T));
else
    N = length(T);
end

% this is irrelevant, but needs to be set
options.K = 2; options.order = 0; 

% is this going to be using the stochastic learning scheme? 
%stochastic_learn = isfield(options,'BIGNbatch') && ...
%    (options.BIGNbatch < N && options.BIGNbatch > 0);
stochastic_learn = iscell(T);
if stochastic_learn, options.BIGNbatch = 2; end
options = checkspelling(options);

% do some data checking and preparation
if xor(iscell(data),iscell(T)), error('X and T must be cells, either both or none of them.'); end
if stochastic_learn % data is a cell, either with strings or with matrices
    if isstruct(data) 
        if isfield(data,'C')
            warning(['The use of semisupervised learning is not implemented for stochatic inference; ' ...
                'removing data.C'])
        end
        data = data.X;
    end
    if ~iscell(data) % make it cell
       dat = cell(N,1); TT = cell(N,1);
       for i=1:N
          t = 1:T(i);
          dat{i} = data(t,:); TT{i} = T(i);
          try data(t,:) = []; 
          catch, error('The dimension of data does not correspond to T');
          end
       end
       if ~isempty(data) 
           error('The dimension of data does not correspond to T');
       end 
       data = dat; T = TT; clear dat TT
    end
else % data can be a cell or a matrix
    if iscell(T)
        for i = 1:length(T)
            if size(T{i},1)==1, T{i} = T{i}'; end
        end
        if size(T,1)==1, T = T'; end
        T = cell2mat(T);
    end
    checkdatacell;
end
[options,data] = checkoptions(options,data,T,0);

if stochastic_learn
    
    % get PCA pre-embedded loadings
    if length(options.pca_spatial) > 1 || (options.pca_spatial > 0 && options.pca_spatial ~= 1)
        if ~isfield(options,'As')
            options.As = highdim_pca(data,T,options.pca_spatial,...
                0,options.standardise,...
                options.onpower,0,options.detrend,...
                options.filter,options.leakagecorr,options.Fs);
        end
        options.pca_spatial = size(options.As,2);
    else
        options.As = [];
    end    
    % main PCA
    [~,~,e_group,e_subj] = highdim_pca(data,T,[],...
        options.embeddedlags,options.standardise,...
        options.onpower,options.varimax,options.detrend,...
        options.filter,options.leakagecorr,options.Fs,options.As);
    
else
    
    % Standardise data and control for ackward trials
    data = standardisedata(data,T,options.standardise);
    % Filtering
    if ~isempty(options.filter)
        data = filterdata(data,T,options.Fs,options.filter);
    end
    % Detrend data
    if options.detrend
        data = detrenddata(data,T);
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
    % main PCA
    [~,~,e_group,e_subj] = highdim_pca(data.X,T,[],0,0,0,options.varimax);

end

end
