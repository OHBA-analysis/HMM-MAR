function [data,T,options] = hmmpreprocess (data,T,options)
% Preprocess the data as the hmmmar function would do, returning the
% preprocessed data and the options struct without these options such that,
% when hmmmar is subsequently called, the preprocessing is not run again. 
% 
% Note: If training is stochastic, it just performs PCA and store the projection
% matrix in options to be used later by hmmmar. 
%
% INPUT
% data          observations, either a struct with X (time series) and C (classes, optional)
%                             or just a matrix containing the time series
% T             length of series
% options       structure with the training options - see documentation in 
%                       https://github.com/OHBA-analysis/HMM-MAR/wiki
%
% OUTPUT
% data          Preprocessed data
% T             length of series, adjusted by the preprocessing
% options       structure with the training options, adjusted so that
%               things are not unnecessarily repeated
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2017)

if iscell(T)
    if size(T,1)==1, T = T'; end
    for i = 1:length(T)
        if size(T{i},1)==1, T{i} = T{i}'; end
    end
    N = numel(cell2mat(T));
else
    N = length(T);
end

% is this going to be using the stochastic learning scheme? 
stochastic_learn = isfield(options,'BIGNbatch') && ...
    (options.BIGNbatch < N && options.BIGNbatch > 0);
options = checkspelling(options);
if ~stochastic_learn && ...
        (isfield(options,'BIGNinitbatch') || ...
        isfield(options,'BIGprior') || ...
        isfield(options,'BIGcyc') || ...
        isfield(options,'BIGmincyc') || ...
        isfield(options,'BIGundertol_tostop') || ...
        isfield(options,'BIGcycnobetter_tostop') || ...
        isfield(options,'BIGtol') || ...
        isfield(options,'BIGinitrep') || ...
        isfield(options,'BIGforgetrate') || ...
        isfield(options,'BIGdelay') || ...
        isfield(options,'BIGbase_weights') || ...
        isfield(options,'BIGcomputeGamma') || ...
        isfield(options,'BIGdecodeGamma') || ...
        isfield(options,'BIGverbose'))
    warning('In order to use stochastic learning, BIGNbatch needs to be specified')
end

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
    if ~iscell(data)
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

% PCA is performed and stored in options; T and data are unmodified
if stochastic_learn 

    error('Not implemented for stochastic learning')

else % the entire pipeline of preprocessing is applied on data and T, 
     % and all options are removed from options 

    % Filtering
    if ~isempty(options.filter)
       data = filterdata(data,T,options.Fs,options.filter);
       options = rmfield(options,'filter');
    end
    % Detrend data
    if options.detrend
       data = detrenddata(data,T); 
       options = rmfield(options,'detrend');
    end
    % Standardise data and control for ackward trials
    data = standardisedata(data,T,options.standardise); 
    options = rmfield(options,'standardise');
    % Leakage correction
    if options.leakagecorr ~= 0 
        data = leakcorr(data,T,options.leakagecorr);
        options = rmfield(options,'leakagecorr');
    end
    % Hilbert envelope
    if options.onpower
       data = rawsignal2power(data,T); 
       options = rmfield(options,'onpower');
    end
    % Leading Phase Eigenvectors 
    if options.leida
        data = leadingPhEigenvector(data,T);
       options = rmfield(options,'leida');
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
       options = rmfield(options,'pca_spatial');
    end    
    % Embedding
    if length(options.embeddedlags) > 1  
        [data,T] = embeddata(data,T,options.embeddedlags);
        options = rmfield(options,'embeddedlags');
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
        options = rmfield(options,'pca');
    end
    % Downsampling
    if options.downsample > 0 
       [data,T] = downsampledata(data,T,options.downsample,options.Fs); 
       options = rmfield(options,'downsample');
    end
end

options = rmfield(options,'orders');

end