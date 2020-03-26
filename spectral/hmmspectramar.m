function fit = hmmspectramar(data,T,hmm,Gamma,options)
% Get spectral estimates from MAR model
% If 'hmm' is specified, it will interrogate its MAR parameters
% If not, will recompute the MAR using maximum likelihood
%
% INPUT
% X             time series (can be [] if MLestimation = 0)
% T             Number of time points for each time series
% hmm           An hmm-mar structure (optional)
% Gamma         State time course (not used if MLestimation=0)
% options       structure with the training options - see documentation in 
%                       https://github.com/OHBA-analysis/HMM-MAR/wiki
%
%
% OUTPUT
% fit is a list with K elements, each of which contains: 
% fit.state(k).psd     (Nf x ndim x ndim) Power Spectral Density matrix
% fit.state(k).ipsd     (Nf x ndim x ndim) Inverse Power Spectral Density matrix
% fit.state(k).coh     (Nf x ndim x ndim) Coherence matrix
% fit.state(k).pcoh    (Nf x ndim x ndim) Partial Coherence matrix
% fit.state(k).pdc   (Nf x ndim x ndim) Baccala's Partial Directed Coherence
% fit.state(k).phase     (Nf x ndim x ndim) Phase matrix
% fit.state(k).psderr: interval of confidence for the cross-spectral density (2 x Nf x ndim x ndim)
% fit.state(k).coherr: interval of confidence for the coherence (2 x Nf x ndim x ndim)
% fit.state(k).pcoherr: interval of confidence for the partial coherence (2 x Nf x ndim x ndim)
% fit.state(k).pdcerr: interval of confidence for the partial directed coherence (2 x Nf x ndim x ndim)
% fit.state(k).f     (Nf x 1) Frequency vector
%       (where ndim is the number of channels) 
% If options.level is 'subject', it also contain psdc, cohc, pcohc, phasec and pdcc with
%           subject specific estimations; their size is (Nf x ndim x ndim x Nsubj)
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2014)

if nargin < 5, options = struct(); end
if nargin < 4, Gamma = []; end
if nargin < 3, hmm = []; end

MLestimation = isempty(hmm) || ~isfield(hmm.state(1),'W') || isempty(hmm.state(1).W.Mu_W);

if MLestimation && ~isempty(options) && isempty(Gamma) && options.K==1
    Gamma = ones(size(data,1),1);
end

if MLestimation && isempty(Gamma)
    error('If the MAR is going to be re-estimated, you need to supply Gamma')
end
if MLestimation && isempty(data)
    error('If the MAR is going to be re-estimated, you need to supply data')
end
if ~MLestimation && isfield(options,'p') && options.p > 0
    warning('Jackknifing is only possible if you re-estimate the MAR models, by leaving hmm empty')
    options.p = 0; 
end
if MLestimation && isfield(options,'zeromean') && options.zeromean == 0
   warning('The use of zeromean=0 is unrecommended here - better restart and set options.zeromean=1') 
   pause(2)
end
if ~isfield(options,'lowrank'), options.lowrank = 0; end

% Adjust options of the hmm
if MLestimation
    hmm = struct('train',struct()); 
    K = size(Gamma,2);   
    if ~isfield(options,'order')
        error('You need to specify options.order')
    end
    if options.order == 0 
        error('order needs to be higher than 0')
    end
    if isfield(options,'embeddedlags') && length(options.embeddedlags)>1
        warning('The options embeddedlags will be ignored')
        options = rmfield(options,'embeddedlags');
    end
    hmm.train.order = options.order; 
    hmm.train.maxorder = options.order; 
    hmm.train.uniqueAR = 0;
    hmm.train.covtype = 'diag';
    hmm.train.orderoffset = 0;
    hmm.train.timelag = 1; 
    hmm.train.exptimelag = 0; 
    hmm.train.lowrank = 0; 
    if isfield(options,'zeromean') 
        hmm.train.zeromean = options.zeromean; 
    else
        hmm.train.zeromean = 1; 
    end
    for k = 1:K, hmm.state(k) = struct('W',struct('Mu_W',[])); end
    order = options.order;
    if order == 0
        error('MAR spectra cannot be estimated for MAR order equal to 0')
    end    
    options.K = K; 
else
    if hmm.train.order == 0 
        error('order needs to be higher than 0')
    end
    if isfield(hmm.state(1),'W')
        ndim = size(hmm.state(1).W.Mu_W,2);
    else
        ndim = length(hmm.state(1).Omega.Gam_rate);
    end
    if isfield(hmm.train,'S') && size(hmm.train.S,1)~=ndim
        hmm.train.S = ones(ndim);
    end
    K = length(hmm.state); 
    if isfield(options,'order') && (options.order ~= hmm.train.maxorder)
        warning('An hmm structure has been specified, so options.order has no effect')
    end
    options.K = length(hmm.state);
    options.order = hmm.train.order;
end

if MLestimation
    if xor(iscell(data),iscell(T))
        error('X and T must be cells, either both or none of them.'); 
    end
    % Adjust T 
    if size(T,1)==1, T = T'; end
    if iscell(data)
        TT = [];
        for j=1:length(T)
            if size(T{j},1)==1, T{j} = T{j}'; end
            t = double(T{j});
            TT = [TT; t];
        end
        T = TT; clear TT
    else
        T = double(T);
    end
    % Load all the data if it is in a cell
    if iscell(data)
        if ~ischar(data{1})
            data = cell2mat(data);
        else
            files = data;
            for j = 1:length(files)
                if ischar(files{j})
                    fsub = files{j};
                    loadfile_sub;
                else
                    X = files{j};
                end
                if j==1
                    data = zeros(sum(T),size(X,2));
                    acc = 0;
                end
                data((1:size(X,1)) + acc, :) = X; acc = acc + size(X,1);
            end
        end
    end
    % Check options
    [options,~,ndim] = checkoptions_spectra(options,data,T,0);
    hmm.train.S = options.S;
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
    % Leakage correction
    if options.leakagecorr ~= 0
        data = leakcorr(data,T,options.leakagecorr);
    end
    % Downsampling
    if options.downsample > 0
        [data,T] = downsampledata(data,T,options.downsample,options.Fs);
    end
    % adjust the dimension of the data
    supposed_order = (size(data,1) - size(Gamma,1) ) / length(T);
    if supposed_order > options.order % trim X
        d = supposed_order-options.order;
        X2 = zeros(sum(T)-length(T)*d,ndim);
        T2 = T - d;
        for in = 1:length(T)
            ind1 = sum(T(1:in-1)) + ( (d+1):T(in) );
            ind2 = sum(T2(1:in-1)) + (1:T2(in));
            X2(ind2,:) = data(ind1,:);
        end
        data = X2; T = T2; clear T2 X2
    elseif supposed_order < options.order % trim Gamma
        d = options.order-supposed_order;
        Gamma2 = zeros(sum(T)-length(T)*options.order,K);
        for in = 1:length(T)
            ind1 = sum(T(1:in-1)) - supposed_order*(in-1) + ( (d+1):(T(in)-supposed_order) );
            ind2 = sum(T(1:in-1)) - options.order*(in-1) + ( 1:(T(in)-options.order) );
            Gamma2(ind2,:) = Gamma(ind1,:);
        end
        Gamma = Gamma2; clear Gamma2
    end
else
    options = checkoptions_spectra(options,[],T,0);
end

if isfield(options,'downsample') && options.downsample~=0
    Fs = options.downsample;
else
    Fs = options.Fs;
end

if options.p>0 && length(T)<5  
    error('You need at least 5 trials to compute error bars for MAR spectra'); 
end

freqs = (0:options.Nf-1)* ...
    ( (options.fpass(2) - options.fpass(1)) / (options.Nf-1)) + options.fpass(1);
w = 2*pi*freqs/Fs;
N = length(T);

% set up the resulting matrices
if options.p==0, NN = 1; 
else, NN = N;
end
Gammasum = zeros(NN,K);
psdc = zeros(options.Nf,ndim,ndim,NN,K);
if ndim>1
    pdcc = zeros(options.Nf,ndim,ndim,NN,K);
end

if MLestimation
    hmm0 = hmm;
end

% Compute the PSD and PDC
for j=1:NN
    % re-estimate the MAR models
    if MLestimation
        if options.p==0 
            Gammaj = Gamma; 
            Xj = data; 
            Tj = T;
        else % subject level estimation
            t0 = sum(T(1:j-1));
            Xj = data(t0+1:t0+T(j),:); Tj = T(j);
            t0 = sum(T(1:j-1)) - (j-1)*hmm.train.maxorder;
            Gammaj = Gamma(t0+1:t0+T(j)-hmm.train.maxorder,:);
            Gammasum(j,:) = sum(Gammaj);
        end
        hmm = mlhmmmar(Xj,Tj,hmm0,Gammaj,options.completelags);
    end
    
    for k = 1:K
        setstateoptions;
        W = zeros(length(orders),ndim,ndim);
        for i = 1:length(orders)
            W(i,:,:) = hmm.state(k).W.Mu_W(~train.zeromean + ((1:ndim) + (i-1)*ndim),:);
        end
        switch train.covtype
            case 'uniquediag'
                covmk = diag(hmm.Omega.Gam_rate / hmm.Omega.Gam_shape);
                preck = diag(hmm.Omega.Gam_shape ./ hmm.Omega.Gam_rate);
                preckd = hmm.Omega.Gam_shape ./ hmm.Omega.Gam_rate;
            case 'diag'
                covmk = diag(hmm.state(k).Omega.Gam_rate / hmm.state(k).Omega.Gam_shape);
                preck = diag(hmm.state(k).Omega.Gam_shape ./ hmm.state(k).Omega.Gam_rate);
                preckd = hmm.state(k).Omega.Gam_shape ./ hmm.state(k).Omega.Gam_rate;
            case 'uniquefull'
                covmk = hmm.Omega.Gam_rate ./ hmm.Omega.Gam_shape;
                preck = inv(covmk);
                preckd = hmm.Omega.Gam_shape ./ diag(hmm.Omega.Gam_rate)';
            case 'full'
                covmk = hmm.state(k).Omega.Gam_rate ./ hmm.state(k).Omega.Gam_shape;
                preck = inv(covmk);
                preckd = hmm.state(k).Omega.Gam_shape ./ diag(hmm.state(k).Omega.Gam_rate)';
        end
        % Get Power Spectral Density matrix and PDC for state K
        for ff = 1:options.Nf
            A = zeros(ndim);
            for i = 1:length(orders)
                o = orders(i);
                A = A + permute(W(i,:,:),[2 3 1]) * exp(-1i*w(ff)*o);
            end
            af_tmp = eye(ndim) - A;
            iaf_tmp = inv(af_tmp); % transfer matrix H
            psdc(ff,:,:,j,k) = iaf_tmp * covmk * iaf_tmp'; 
            % Get PDC
            if options.to_do(2)==1 && ndim>1
                for n = 1:ndim
                    for l = 1:ndim
                        pdcc(ff,n,l,j,k) = sqrt(preckd(n)) * abs(af_tmp(n,l)) / ...
                            sqrt( preckd * (abs(af_tmp(:,l)).^2) );
                    end
                end
            end
        end
    end
end

% Normalise psdc and pdcc by the amount of fractional occupancy per trial
if options.p > 0
    for ff = 1:options.Nf
        for n = 1:ndim
            for l = 1:ndim
                psdc(ff,n,l,:,:) = permute(psdc(ff,n,l,:,:),[4 5 1 2 3]) ./ Gammasum;
                pdcc(ff,n,l,:,:) = permute(pdcc(ff,n,l,:,:),[4 5 1 2 3]) ./ Gammasum;
            end
        end
    end
end
    
% For each state compute coherence, and do jackknifing
for k = 1:K
    fit.state(k).pdc = []; fit.state(k).coh = []; fit.state(k).pcoh = []; fit.state(k).phase = [];
    if options.p > 0
        fit.state(k).psd = sum(psdc(:,:,:,:,k),4) / sum(Gammasum(:,k));
    else
        fit.state(k).psd = psdc(:,:,:,1,k);
    end
    if options.to_do(2)==1  && ndim>1
        if options.p > 0
            fit.state(k).pdc = sum(pdcc(:,:,:,:,k),4) / sum(Gammasum(:,k));
        else
            fit.state(k).pdc = pdcc(:,:,:,1,k);
        end
    end
    fit.state(k).ipsd = zeros(options.Nf,ndim,ndim);
    for ff = 1:options.Nf
        fit.state(k).ipsd(ff,:,:) = inv(permute(fit.state(k).psd(ff,:,:),[3 2 1]));
    end
    fit.state(k).f = freqs;
    
    % Get Coherence and Phase
    if options.to_do(1)==1 && ndim>1
        fit.state(k).coh = zeros(options.Nf,ndim,ndim);
        fit.state(k).pcoh = zeros(options.Nf,ndim,ndim);
        fit.state(k).phase = zeros(options.Nf,ndim,ndim);
        for n = 1:ndim
            for l = 1:ndim
                rkj=fit.state(k).psd(:,n,l)./(sqrt(fit.state(k).psd(:,n,n)).*sqrt(fit.state(k).psd(:,l,l)));
                fit.state(k).coh(:,n,l)=abs(rkj);
                fit.state(k).pcoh(:,n,l)=-fit.state(k).ipsd(:,n,l)./...
                    (sqrt(fit.state(k).ipsd(:,n,n)).*sqrt(fit.state(k).ipsd(:,l,l)));
                fit.state(k).phase(:,n,l)=atan(imag(rkj)./real(rkj));
            end
        end
    end
    
    if options.p > 0 % jackknife
        [psderr,coherr,pcoherr,pdcerr,sdphase] = ...
            spectrerr(psdc(:,:,:,:,k),pdcc(:,:,:,:,k),fit.state(k).coh, ...
            fit.state(k).pcoh,fit.state(k).pdc,options,1);
        fit.state(k).psderr = psderr;
        if options.to_do(1)==1 && ndim>1
            fit.state(k).coherr = coherr;
            fit.state(k).pcoherr = pcoherr;
            fit.state(k).sdphase = sdphase;
        end
        if options.to_do(2)==1 && ndim>1
            fit.state(k).pdcerr = pdcerr;
        end
    end
    
    % weight the PSD by the inverse of the sampling rate
    fit.state(k).psd = (1/Fs) * fit.state(k).psd;
    % and take abs value for the diagonal
    for n=1:ndim, fit.state(k).psd(:,n,n) = abs(fit.state(k).psd(:,n,n)); end
    if options.p > 0
        fit.state(k).psderr = (1/Fs) * fit.state(k).psderr;  
        for n=1:ndim
            fit.state(k).psderr(1,:,n,n) = abs(fit.state(k).psderr(1,:,n,n));
            fit.state(k).psderr(2,:,n,n) = abs(fit.state(k).psderr(2,:,n,n));
        end
    end

end
end

