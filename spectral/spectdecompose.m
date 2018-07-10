function [sp_fit,sp_fit_group,sp_profiles] = spectdecompose(sp_fit,options)
%
% From a multitaper or MAR estimation of the states spectra (per subject),  
% it factorises the spectral information (which is given by frequency bin)
% into a set of spectral components (which are much fewer than the
% frequency bins), in order to facilitate interpretation and visualisation
% The decomposition can be done either using PCA or non-negative matrix
% factorisation or NNMF (the default). Note that if NNMF is used, the
% solution might somewhat vary every time the function is called. 
%
% INPUTS: 
%
% sp_fit                The output of hmmspectramar or hmmspectramt. It
%                       should be a cell, where each element corresponds 
%                       to one subject
% options               Struct indicating the decomposing options
%        .Ncomp         Number of components (default 4)
%        .Method        'NNMF' or 'PCA' (default 'NNMF')
%        .Base          What to base the factorisation on: PSD if 'psd' or on
%                       coherence if 'coh' (default: 'coh') 
%        .sp_profiles   Spectral profiles; if supplied, these will be used
%                       instead of computing them (default: empty) 
%        .plot          If the spectral profiles are to be plotted (default: 0) 
% 
% OUTPUT:
% 
% sp_fit                A cell with the spectral components, with the same fields  
%                       and dimensions that the input 'sp_fit', but with Ncomp 
%                       components instead of No. frequency bins. If there
%                       is only one subject, then sp_field is a single struct
%                 
% sp_fit_group          A single struct containing the mean across subjects
%                       of sp_fit. If there is only one subject, then
%                       sp_fit_group equals sp_field
% sp_profiles           The (frequency bins by spectral components) mixing
%                       matrix used to project from (no. of frequency bins
%                       by regions) to (no. of components by regions)
%
% Author: Mark Woolrich, OHBA, University of Oxford (2017)
%         Diego Vidaurre, OHBA, University of Oxford (2017)

if ~iscell(sp_fit)
    error('Variable fit needs to be a cell, with one estimation per subject')
end

if nargin < 2, options = struct(); end
if ~isfield(options,'Ncomp'), options.Ncomp = 4; end
if ~isfield(options,'Method'), options.Method = 'NNMF'; end
if ~isfield(options,'Base'), options.Base = 'coh'; end
if ~isfield(options,'plot'), options.plot = 0; end

ndim = size(sp_fit{1}.state(1).psd,2); % no. channels
ndim2 = ndim*(ndim-1)/2;
Nf = size(sp_fit{1}.state(1).psd,1); % no. frequencies
N = length(sp_fit); % no. subjects
K = length(sp_fit{1}.state); % no. states
ind_offdiag = triu(true(ndim),1)==1;

% check that all estimations have the same number of freq bins
for n = 1:N
    if isempty(sp_fit{n})
        error('One or more elements of sp_fit are empty - remove them first')
    end
    if size(sp_fit{n}.state(1).psd,1) ~= Nf
        error(['It is necessary for the spectral estimation of all subjects ' ...
            'to have the same number of frequency bins. ' ...
            'In the case of the multitaper, this is can be done by setting ' ...
            'the options tapers and win to a fixed value.'])
    end
end

% put coh and psd in temporary arrays
coh_comps = zeros(N,K,Nf,ndim,ndim);
psd_comps = zeros(N,K,Nf,ndim);
for n = 1:N
    for k = 1:K
        psd = sp_fit{n}.state(k).psd;
        coh = sp_fit{n}.state(k).coh;
        for j = 1:ndim
            psd_comps(n,k,:,j) = psd(:,j,j);
            for l=1:ndim
                coh_comps(n,k,:,j,l) = coh(:,j,l);
            end
        end
    end
end

% Build the matrix that is to be factorised
Xpsd = zeros(Nf,ndim*K);
for k=1:K
    ind = (1:ndim) + (k-1)*ndim;
    Xpsd(:,ind)= squeeze(mean(abs(psd_comps(:,k,:,:)),1));
end
Xcoh = zeros(Nf,K*ndim2);
for k = 1:K
    ind = (1:ndim2) + (k-1)*ndim2;
    ck = squeeze(mean(abs(coh_comps(:,k,:,:,:)),1));
    Xcoh(:,ind) = ck(:,ind_offdiag);
end
if strcmpi(options.Base,'psd')
    X = Xpsd;
else
    X = Xcoh;
end

if ~isfield(options,'sp_profiles') || isempty(options.sp_profiles)
    % Doing the decomposition
    if strcmpi(options.Method,'NNMF')
        try
            [a,~] = nnmf(X,options.Ncomp,'replicates',500,'algorithm','als');
        catch
            error('nnmf not found - perhaps the Matlab version is too old')
        end
        %sp_profiles = pinv(X') * b'; % you lose non-negativity by doing this
        sp_profiles = a;
        [~,ind] = max(sp_profiles',[],2);
        [~,neworder_auto] = sort(ind);
        sp_profiles = sp_profiles(:,neworder_auto);
    else
        try
            m = mean(X);
            X = X - repmat(m,Nf,1);
            [b,a] = pca(X,'NumComponents',options.Ncomp);
        catch
            error('Error running pca - maybe not matlab''s own?')
        end
        sp_profiles = a;
    end
else
    sp_profiles = options.sp_profiles;
end

% plot if required
if options.plot
    figure;
    if options.Ncomp > 4
        j1 = ceil(options.Ncomp/2); j2 = 2;
    else
        j1 = options.Ncomp; j2 = 1;
    end
    for j = 1:options.Ncomp
        subplot(j1,j2,j)
        plot(sp_profiles(:,j),'LineWidth',2.5)
    end
end

% group level
sp_fit_group = struct();
sp_fit_group.state = struct();
% NNMF / PCA
if strcmpi(options.Method,'NNMF')
    opt = statset('maxiter',1);
    [~,b] = nnmf(Xpsd,options.Ncomp,'algorithm','als',...
        'w0',sp_profiles,'Options',opt);
    psd = b'; % regions by components
    [~,b] = nnmf(Xcoh,options.Ncomp,'algorithm','als',...
        'w0',sp_profiles,'Options',opt);
    coh = b'; % pairs of regions by components
else
    Xpsd = Xpsd - repmat(mean(Xpsd),Nf,1);
    Xcoh = Xcoh - repmat(mean(Xcoh),Nf,1);
    psd = (Xpsd' * sp_profiles); % ndim by components
    coh = (Xcoh' * sp_profiles);
end
for k = 1:K
    sp_fit_group.state(k).psd = zeros(options.Ncomp,ndim,ndim);
    sp_fit_group.state(k).coh = ones(options.Ncomp,ndim,ndim);
    ind = (1:ndim) + (k-1)*ndim;
    for i = 1:options.Ncomp
        sp_fit_group.state(k).psd(i,:,:) = diag(psd(ind,i));
    end
    ind = (1:ndim2) + (k-1)*ndim2;
    for i = 1:options.Ncomp
        graphmat = zeros(ndim);
        graphmat(ind_offdiag) = coh(ind,i);
        graphmat=(graphmat+graphmat') + eye(ndim);
        sp_fit_group.state(k).coh(i,:,:) = graphmat;
    end
end

% Subject level
if N>1
    for n = 1:N
        sp_fit{n} = struct();
        sp_fit{n}.state = struct();
        % prepare matrix
        Xpsd = zeros(Nf,ndim*K);
        for k=1:K
            ind = (1:ndim) + (k-1)*ndim;
            Xpsd(:,ind)= squeeze(abs(psd_comps(n,k,:,:)));
        end
        Xcoh = zeros(Nf,K*ndim2);
        for k = 1:K
            ind = (1:ndim2) + (k-1)*ndim2;
            ck = squeeze(abs(coh_comps(n,k,:,:,:)));
            Xcoh(:,ind) = ck(:,ind_offdiag);
        end
        % NNMF / PCA
        if strcmpi(options.Method,'NNMF')
            opt = statset('maxiter',1);
            [~,b] = nnmf(Xpsd,options.Ncomp,'algorithm','als',...
                'w0',sp_profiles,'Options',opt);
            psd = b'; % regions by components
            [~,b] = nnmf(Xcoh,options.Ncomp,'algorithm','als',...
                'w0',sp_profiles,'Options',opt);
            coh = b';
        else
            Xpsd = Xpsd - repmat(mean(Xpsd),Nf,1);
            Xcoh = Xcoh - repmat(mean(Xcoh),Nf,1);
            psd = (Xpsd' * sp_profiles);
            coh = (Xcoh' * sp_profiles);
        end
        % Reshape stuff
        for k = 1:K
            sp_fit{n}.state(k).psd = zeros(options.Ncomp,ndim,ndim);
            sp_fit{n}.state(k).coh = ones(options.Ncomp,ndim,ndim);
            ind = (1:ndim) + (k-1)*ndim;
            for i = 1:options.Ncomp
                sp_fit{n}.state(k).psd(i,:,:) = diag(psd(ind,i));
            end
            ind = (1:ndim2) + (k-1)*ndim2;
            for i = 1:options.Ncomp
                graphmat = zeros(ndim);
                graphmat(ind_offdiag) = coh(ind,i);
                graphmat=(graphmat+graphmat') + eye(ndim);
                sp_fit{n}.state(k).coh(i,:,:) = graphmat;
            end
        end
    end
else
    sp_fit = sp_fit_group;
end

end


