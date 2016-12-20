function hmm = hmmsepinit(data,T,options)
% Compute separate HMMs for each subject, which can be later be used
% to speed up the stochastic HMM initialisation by placing them into options.initial_hmm
% Note: if memory problems arise, manually adjust the number of parallel processors 
% 
% INPUTS
% data: cell with strings referring to the files containing each subject's data, 
%       or cell with with matrices (time points x channels) with each
%       subject's data
% T: cell of vectors, where each element has the length of each trial per
%       subject. Dimension of T{n} has to be (1 x nTrials)
% options: HMM options for both the subject runs
%
% OUTPUT
% hmm:  a list of HMM, one per subject
%
% Diego Vidaurre, OHBA, University of Oxford (2016)

N = length(T); K = options.K;
hmm = cell(N,1);

if ~iscell(data)
    dat = cell(N,1); TT = cell(N,1);
    for i=1:N
        t = 1:T(i);
        dat{i} = data(t,:); TT{i} = T(i);
        try data(t,:) = [];
        catch, error('The dimension of data does not correspond to T');
        end
    end
    if ~isempty(data),
        error('The dimension of data does not correspond to T');
    end
    data = dat; T = TT; clear dat TT
end

% minimal option checking
if ~isfield(options,'order'), error('order was not specified'); end
if ~isfield(options,'pcapred'), options.pcapred = 0; end
if ~isfield(options,'vcomp') && options.pcapred>0, options.vcomp = 1; end
if ~isfield(options,'pcamar'), options.pcamar = 0; end
if ~isfield(options,'pca'), options.pca = 0; end
if ~isfield(options,'timelag'), options.timelag = 1; end
if ~isfield(options,'exptimelag'), options.exptimelag = 1; end
if ~isfield(options,'orderoffset'), options.orderoffset = 0; end
if ~isfield(options,'standardise'), 
    options.standardise = (length(options.pca)>1) || (options.pca>0); 
end
if ~isfield(options,'embeddedlags'), options.embeddedlags = 0; end
options.orders = formorders(options.order,options.orderoffset,options.timelag,options.exptimelag);
options.dropstates = 0;

% get PCA loadings
if length(options.pca) > 1 || options.pca > 0
    if ~isfield(options,'A')
        options.A = highdim_pca(data,T,options.pca,options.embeddedlags,options.standardise);
    end
    options.ndim = size(options.A,2);
end
if options.pcamar > 0 && ~isfield(options,'B')
    options.B = pcamar_decomp(data,T,options);
end
if options.pcapred > 0 && ~isfield(options,'V')
    options.V = pcapred_decomp(data,T,options);
end


options = rmfield(options,'orders');
I = randpermNK(N,options.BIGNinitbatch);

for ii = 1:length(I)
    subset = I{ii};
    [X,~,~,Ti] = loadfile(Xin(subset),T(subset),options);  
    hmm{ii} = hmmmar(X,T{ii},options);
    hmm{ii}.subset = subset; 
    fprintf('Init: batch %d: %d states active \n',ii,sum(hmm{ii}.train.active) )
end

end


