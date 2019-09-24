function [sp_fit_group,sp_profiles,sp_fit] = spectdecompose(sp_fit,options,chan)
%
% From a multitaper or MAR estimation of the states spectra (per subject),  
% it factorises the spectral information (which is given by frequency bin)
% into a set of spectral components (which are much fewer than the
% frequency bins), in order to facilitate interpretation and visualisation.
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
% chan                  If supplied, a vector of integers defining which
%                       channels will be used in the decomposion; 
%                       if not specified, all will beused
% 
% OUTPUT:
%               
% sp_fit_group          A single struct containing the mean across subjects
%                       of sp_fit. If there is only one subject, then
%                       sp_fit_group equals sp_fit
% sp_profiles           The (frequency bins by spectral components) mixing
%                       matrix used to project from (no. of frequency bins
%                       by regions) to (no. of components by regions)
% sp_fit                A cell with the spectral components, with the same fields  
%                       and dimensions that the input 'sp_fit', but with Ncomp 
%                       components instead of No. frequency bins. If there
%                       is only one subject, then sp_fit is a single struct
%   
% Authors: Mark Woolrich, OHBA, University of Oxford (2017)
%          Diego Vidaurre, OHBA, University of Oxford (2017)
%          Andrew Quinn, OHBA, University of Oxford (2018)

if ~iscell(sp_fit)
    warning('Only one subject was provided; recommended to run at the group level')
    aux = sp_fit;
    sp_fit = {};
    sp_fit{1} = aux;   
    clear aux
end

if nargin < 2, options = struct(); end
if ~isfield(options,'Niterations'), options.Niterations = 10; end
if ~isfield(options,'Ncomp'), options.Ncomp = 4; end
if ~isfield(options,'Method'), options.Method = 'NNMF'; end
if ~isfield(options,'Base'), options.Base = 'coh'; end
if ~isfield(options,'plot'), options.plot = 0; end

ndim = size(sp_fit{1}.state(1).psd,2); % no. channels
if nargin < 3, chan = 1:ndim;
else, ndim = length(chan);
end
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
psd_comps = zeros(N,K,Nf,ndim);
coh_comps = zeros(N,K,Nf,ndim,ndim);
for n = 1:N
    for k = 1:K
        psd = sp_fit{n}.state(k).psd(:,chan,chan);
        coh = sp_fit{n}.state(k).coh(:,chan,chan);
        for j = chan
            psd_comps(n,k,:,j) = psd(:,j,j);
            for l = chan
                coh_comps(n,k,:,j,l) = coh(:,j,l);
            end
        end
    end
end

% Build the matrix that is to be factorised
Xpsd = zeros(Nf,ndim*K);
keep_psd = true(1,ndim*K);
for k = 1:K
    ind = (1:ndim) + (k-1)*ndim;
    notnan = ~isnan(psd_comps(:,k,1,1)) & ~isinf(psd_comps(:,k,1,1));
    if any(notnan)
        Xpsd(:,ind)= squeeze(mean(abs(psd_comps(notnan,k,:,:)),1)); % mean across subjects
    else
        keep_psd(ind) = false;
    end
end
Xcoh = zeros(Nf,K*ndim2);
keep_coh = true(1,ndim2*K);
for k = 1:K
    ind = (1:ndim2) + (k-1)*ndim2;
    notnan = ~isnan(coh_comps(:,k,1,1,2)) & ~isinf(coh_comps(:,k,1,1,2));
    if any(notnan)
        ck = squeeze(mean(abs(coh_comps(notnan,k,:,:,:)),1));
        Xcoh(:,ind) = ck(:,ind_offdiag);
    else
        Xcoh(:,ind) = false;
    end
    
end
if strcmpi(options.Base,'psd')
    X = Xpsd(:,keep_psd);
else
    X = Xcoh(:,keep_coh);
end

% Specify fit function, a unimodal gaussian
gauss_func = @(x,f) f.a1.*exp(-((x-f.b1)/f.c1).^2);
% Default fit options
options_fit = fitoptions('gauss1');
% constrain lower and upper bounds
options_fit.Lower = [0,1,0];
options_fit.Upper = [Inf,size(X,1),size(X,1)];

if ~isfield(options,'sp_profiles') || isempty(options.sp_profiles)
    % Doing the decomposition
    if strcmpi(options.Method,'NNMF')
        bestval = Inf; fitval = zeros(options.Niterations,1); 
        for ii = 1:options.Niterations
            try
                [A,~] = nnmf(X,options.Ncomp,'replicates',500,'algorithm','als');
            catch
                error('nnmf not found - perhaps the Matlab version is too old')
            end 
            for jj = 1:size(A,2)
                f = fit( linspace(1,size(A,1),size(A,1))',A(:,jj), 'gauss1',options_fit);
                residuals = A(:,jj) - gauss_func(1:size(A,1),f)';
                fitval(ii) = fitval(ii) + sum( residuals.^2 ) / size(A,2);
            end
            if fitval(ii) < bestval
                bestval = fitval(ii);
                bestfit = A; 
            end
        end
        %sp_profiles = pinv(X') * b'; % you lose non-negativity by doing this
        sp_profiles = bestfit;
        [~,ind] = max(sp_profiles,[],1);
        [~,neworder_auto] = sort(ind);
        sp_profiles = sp_profiles(:,neworder_auto);
    else
        try
            m = mean(X);
            X = X - repmat(m,Nf,1);
            [~,A] = pca(X,'NumComponents',options.Ncomp);
        catch
            error('Error running pca - maybe not matlab''s own?')
        end
        sp_profiles = A;
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
psd = zeros(ndim*K,options.Ncomp);
coh = zeros(ndim2*K,options.Ncomp);
if strcmpi(options.Method,'NNMF')
   psd(keep_psd,:) = (Xpsd(:,keep_psd)' * sp_profiles); % ndim by components
   coh(keep_coh,:) = (Xcoh(:,keep_coh)' * sp_profiles);
%     opt = statset('maxiter',0);
%     [~,b] = nnmf(Xpsd,options.Ncomp,'algorithm','als',...
%         'w0',sp_profiles,'Options',opt);
%     psd = b'; % regions by components
%     [~,b] = nnmf(Xcoh,options.Ncomp,'algorithm','als',...
%         'w0',sp_profiles,'Options',opt);
%     coh = b'; % pairs of regions by components
else
    Xpsd(:,keep_psd) = Xpsd(:,keep_psd) - repmat(mean(Xpsd(:,keep_psd)),Nf,1);
    Xcoh(:,keep_coh) = Xcoh(:,keep_coh) - repmat(mean(Xcoh(:,keep_coh)),Nf,1);
    psd(keep_psd,:) = (Xpsd(:,keep_psd)' * sp_profiles); % ndim by components
    coh(keep_coh,:) = (Xcoh(:,keep_coh)' * sp_profiles);
end
if any(isnan(psd(:))) || any(isnan(coh(:))) 
   warning('There are NaNs in the estimations for those states that were not used') 
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
if N > 1 && nargout == 3
    for n = 1:N
        sp_fit{n} = struct();
        sp_fit{n}.state = struct();
        % prepare matrix
        Xpsd = zeros(Nf,ndim*K);
        keep_psd = true(1,ndim*K);
        for k = 1:K
            ind = (1:ndim) + (k-1)*ndim;
            Xpsd(:,ind)= squeeze(abs(psd_comps(n,k,:,:)));
            if any(isnan(var(Xpsd(:,ind)))) || any(isinf(var(Xpsd(:,ind))))
                keep_psd(ind) = false;
                warning(['Session ' num2str(n) ' did not use state ' num2str(k) '; PSD set to NaN'])
            end
        end
        Xcoh = zeros(Nf,K*ndim2);
        keep_coh = true(1,K*ndim2);
        for k = 1:K
            ind = (1:ndim2) + (k-1)*ndim2;
            ck = squeeze(abs(coh_comps(n,k,:,:,:)));
            Xcoh(:,ind) = ck(:,ind_offdiag);
            if any(isnan(var(Xcoh(:,ind)))) || any(isinf(var(Xcoh(:,ind))))
                keep_coh(ind) = false; 
                warning(['Session ' num2str(n) ' did not use state ' num2str(k) '; Coh set to NaN'])
            end
        end
        % NNMF / PCA
        psd = zeros(ndim*K,options.Ncomp);
        coh = zeros(ndim2*K,options.Ncomp);
        if strcmpi(options.Method,'NNMF')
            opt = statset('maxiter',1);
            [~,b] = nnmf(Xpsd(:,keep_psd),options.Ncomp,'algorithm','als',...
                'w0',sp_profiles,'Options',opt);
            psd(keep_psd,:) = b'; % regions by components
            [~,b] = nnmf(Xcoh(:,keep_coh),options.Ncomp,'algorithm','als',...
                'w0',sp_profiles,'Options',opt);
            coh(keep_coh,:) = b';
        else
            Xpsd(:,keep_psd) = Xpsd(:,keep_psd) - repmat(mean(Xpsd(:,keep_psd)),Nf,1);
            Xcoh(:,keep_coh) = Xcoh(:,keep_coh) - repmat(mean(Xcoh(:,keep_coh)),Nf,1);
            psd(keep_psd,:) = (Xpsd(:,keep_psd)' * sp_profiles);
            coh(keep_coh,:) = (Xcoh(:,keep_coh)' * sp_profiles);
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


