function [tuda,Gamma,encmodel,decmodel] = tudasparsify(X,Y,T,tuda,...
    Gamma,temp_sparsity,spatial_sparsity,doplot)
% Given an estimation of a TUDA model and the corresponding model time
% courses Gamma, this function creates sparse estimates of the decoding models.
% This is done:
% - Temporally: after fitting a Gaussian distribution to the time point of maximal 
%   occupancy for each state, anything out of the center of mass of this
%   Gaussian disitribution is made zero; that is, those visits to the
%   decoders that are far away from the time point of maximal occupancy are
%   eliminated. This is controlled by parameter temp_sparsity; for example,
%   for temp_sparsity=0.1, we'll be getting rid of everything out of the 90% 
%   area of probability density of the fitted Gaussian distribution. For a
%   visualisation of what it does, set doplot = 1.
% - Spatially: Using only the time points dictated by the temporal sparsity
%   constraint, the decoding models are re-estimated with a elastic-net
%   sparsity constraint, so that some decoding coefficients are driven to
%   zero; this is controlled by the spatial_sparsity parameters, a value 
%   between 0 and 1 that determines the proportion of coefficients to
%   eliminate. For example, if spatial_sparsity=0.25, 1/4 of the decoding
%   coefficients will be made zero. 
%
% The output also includes the resulting encoding models (as obtained from
%   the tudaencoding function) and the sparse decoding models in a 
%   (channel by stimuli by number of decoders) matrix
% 
% Author: Diego Vidaurre, OHBA, University of Oxford (2019)

if nargin < 6, temp_sparsity = 0.1; end
if nargin < 7, spatial_sparsity = 0.5; end
if nargin < 8, doplot = true; end

max_num_classes = 5;
do_preproc = 1; 

N = length(T); ttrial = T(1); q = size(Y,2); K = tuda.train.K;

% Check options and put data in the right format
tuda.train.parallel_trials = 0; 
if isfield(tuda.train,'orders')
    orders = tuda.train.orders;
    tuda.train = rmfield(tuda.train,'orders');
else
    orders = [];
end
if isfield(tuda.train,'active')
    active = tuda.train.active;
    tuda.train = rmfield(tuda.train,'active');
else
    active = [];
end

classification = length(unique(Y(:))) < max_num_classes;
if classification
    % no demeaning by default if this is a classification problem
    if ~isfield(tuda.train,'demeanstim'), tuda.train.demeanstim = 0; end
end

if do_preproc
    [X,Y,T,options] = preproc4hmm(X,Y,T,tuda.train); % this demeans Y
    p = size(X,2);
end
if ~isempty(active), tuda.train.active = active; end 
if ~isempty(orders),  tuda.train.orders = orders;  end 

mg = getFractionalOccupancy(Gamma,T,tuda.train,1);
mg_sparse = zeros(size(mg));

% Temporal sparsification
for k = 1:K
    % Specify fit function, a unimodal gaussian
    gauss_func = @(x,f) f.a1.*exp(-((x-f.b1)/f.c1).^2);
    options_fit = fitoptions('gauss1');
    [~,m] = max(mg(:,k));
    options_fit.Lower = [0,m,0];
    options_fit.Upper = [Inf,m,ttrial];
    f = fit( linspace(1,ttrial,ttrial)',mg(:,k), 'gauss1',options_fit);
    mg_sparse(:,k) = gauss_func(1:ttrial,f)';
    mu = f.b1; s = f.c1;
    if temp_sparsity > 0
        pp = temp_sparsity / 2;
        d = norminv(pp,mu,s);
        if d >= 1, mg_sparse(1:round(d),k) = 0; end
        pp = 1 - temp_sparsity / 2;
        d = norminv(pp,mu,s);
        if d <= ttrial, mg_sparse(round(d):ttrial,k) = 0; end
    end
end

for j = 1:N
    ind = (1:T(j)) + sum(T(1:j-1));
    for k = 1:K
       ind_zero =  mg_sparse(:,k) == 0;
       Gamma(ind(ind_zero),k) = 0; 
    end
end

Gamma = rdiv(Gamma,sum(Gamma,2));

% Spatial sparsification
for k = 1:K
   ind = Gamma(:,k) > 0;
   beta = lassoglm(X(ind,:),Y(ind,:),'normal','Weights',Gamma(ind,k),...
       'Alpha',0.1,'DFmax',round(p*(1-spatial_sparsity)));
   tuda.state(k).W.Mu_W(1:end-1,end-q+1:end) = beta(:,1); 
end

decmodel = tudabeta(tuda); % channels by states
encmodel = tudaencoding(X,Y,T,options,Gamma); % channels by states

if doplot
    figure
    subplot(211)
    plot(mg,'LineWidth',3); xlim([1 size(mg,1)])
    title('Average decoder occupancy')
    set(gca,'FontSize',16)
    subplot(212)
    plot(mg_sparse,'LineWidth',3); xlim([1 size(mg_sparse,1)])
    title('Fitted Gaussian distributions')
    set(gca,'FontSize',16)
    figure
    subplot(121)
    imagesc(decmodel);colorbar
    title('Sparse decoding models'); ylabel('Sensors'); xlabel('Decoders')
    set(gca,'FontSize',16)
    subplot(122)
    imagesc(encmodel);colorbar
    title('Sparse encoding models'); ylabel('Sensors'); xlabel('Encoders')
    set(gca,'FontSize',16)   
end

end


