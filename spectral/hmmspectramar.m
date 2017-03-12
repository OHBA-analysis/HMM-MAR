function fit = hmmspectramar(X,T,hmm,Gamma,options)
% Get ML spectral estimates from MAR model
%
% INPUT
% X             time series (can be [] if options.MLestimation = 0)
% T             Number of time points for each time series
% hmm           An hmm-mar structure (optional)
% Gamma         State time course (not used if options.MLestimation=0)
% options 

%  .Fs:       Sampling frequency
%  .fpass:    Frequency band to be used [fmin fmax] (default [0 Fs/2])
%  .p:        p-value for computing jackknife confidence intervals (default 0)
%  .Nf        No. of frequencies to be computed in the range minHz-maxHz
%  .order     If we want a higher MAR order than that used for training,
%               specify it here - a new set of MAR models will be estimated
%  .MLestimation     if 0, it will use the
%               MAR models as they were returned by the HMM-MAR inference (i.e. a
%               posterior distribution instead of maximum likelihood)
%  .completelags    if MLestimation is true and completelags is true,
%               the MAR spectra will be calculated with the complete set of
%               lags up to the specified order (lags=1,2,...,order)
%  .level       'group' (by default) for group level estimations, or
%               'subject' for subject level (no jackknife is allowed)
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

if ~isfield(options,'MLestimation'), options.MLestimation = 1 ; end

if options.MLestimation && isempty(Gamma)
    error('If MLestimation=1, you need to supply Gamma')
end

if iscell(T)
    for i = 1:length(T)
        if size(T{i},1)==1, T{i} = T{i}'; end
    end
    T0 = T; 
    T = cell2mat(T);
else
    T0 = T; 
end
if iscell(X)
    if size(X,1)==1, X = X'; end
    X = cell2mat(X);
end

if ~isempty(hmm)
    if isfield(hmm.state(1),'W')
        ndim = size(hmm.state(1).W.Mu_W,2);
    else
        ndim = length(hmm.state(1).Omega.Gam_rate);
    end
    if isfield(hmm.train,'S') && size(hmm.train.S,1)~=ndim
        hmm.train.S = ones(ndim);
    end
    K = length(hmm.state); 
    if isfield(options,'order') % a new order?
        warning('An hmm structure has been specified, so options.order has no effect')
    end
    options.order = hmm.train.maxorder;
else
    ndim = size(X,2);
    hmm = struct('train',struct()); hmm.train.S = ones(ndim);
    K = size(Gamma,2);   
    if ~isfield(options,'order')
        options.order = (size(X,1) - size(Gamma,1) ) / length(T);
    end
    hmm.train.order = options.order; 
    hmm.train.maxorder = options.order; 
    hmm.train.multipleConf = 0;
    hmm.train.uniqueAR = 0;
    hmm.train.covtype = 'diag';
    hmm.train.orderoffset = 0;
    hmm.train.timelag = 1; 
    hmm.train.exptimelag = 0; 
    if isfield(options,'zeromean'), 
        hmm.train.zeromean = options.zeromean; 
    else
        hmm.train.zeromean = 1; 
    end
    for k=1:K, hmm.state(k) = struct('W',struct('Mu_W',[])); end
end

options = checkoptions_spectra(options,ndim,T);

if options.MLestimation 
   supposed_order = (size(X,1) - size(Gamma,1) ) / length(T);
   if supposed_order > options.order % trim X
       d = supposed_order-options.order;
       X2 = zeros(sum(T)-length(T)*d,ndim);
       T2 = T - d; 
       for in = 1:length(T),
           ind1 = sum(T(1:in-1)) + ( (d+1):T(in) );
           ind2 = sum(T2(1:in-1)) + (1:T2(in));
           X2(ind2,:) = X(ind1,:); 
       end
       X = X2; T = T2; clear T2 X2
   elseif supposed_order < options.order % trim Gamma
       d = options.order-supposed_order;
       Gamma2 = zeros(sum(T)-length(T)*options.order,K);
       for in = 1:length(T),
           ind1 = sum(T(1:in-1)) - supposed_order*(in-1) + ( (d+1):(T(in)-supposed_order) );
           ind2 = sum(T(1:in-1)) - options.order*(in-1) + ( 1:(T(in)-options.order) );
           Gamma2(ind2,:) = Gamma(ind1,:);
       end
       Gamma = Gamma2; clear Gamma2
   end
   hmm.train.maxorder = options.order; 
end


if hmm.train.maxorder==0
    error('MAR spectra cannot be estimated for MAR order equal to 0')
end

if options.p>0 && length(T)<5  
    error('You need at least 5 trials to compute error bars for MAR spectra'); 
end

%loadings = options.loadings;
%if hmm.train.whitening, loadings = loadings * iA; end
%M = size(options.loadings,1);

freqs = (0:options.Nf-1)* ...
    ( (options.fpass(2) - options.fpass(1)) / (options.Nf-1)) + options.fpass(1);
w = 2*pi*freqs/options.Fs;
N = length(T0);
Gammasum = zeros(N,K);

if options.p==0
    if strcmp(options.level,'group'), 
        NN = 1; 
    else
        NN = N;
        if ndim>1
            cohc = zeros(options.Nf,ndim,ndim,NN,K);
            pcohc = zeros(options.Nf,ndim,ndim,NN,K);
            phasec = zeros(options.Nf,ndim,ndim,NN,K);
        end
    end
    psdc = zeros(options.Nf,ndim,ndim,NN,K);
    if ndim>1
        pdcc = zeros(options.Nf,ndim,ndim,NN,K);
        ipsdc = zeros(options.Nf,ndim,ndim,NN,K);
    end
else % necessarily, options.level is 'group'
    psdc = zeros(options.Nf,ndim,ndim,1,K);
    pdcc = zeros(options.Nf,ndim,ndim,1,K);
    ipsdc = zeros(options.Nf,ndim,ndim,1,K);
    NN = N;
end

if options.MLestimation
    if size(T,1)==1, T=T'; end
    hmm0 = hmm;
    if isfield(hmm0.train,'B'), 
        hmm0.train = rmfield(hmm0.train,'B'); 
    end
    if isfield(hmm0.train,'V'), 
        hmm0.train = rmfield(hmm0.train,'V'); 
    end    
end

for j=1:NN
    
    if options.MLestimation
        if options.p==0 && strcmp(options.level,'group')
            Gammaj = Gamma; Xj = X; Tj = T;
        elseif options.p>0 && strcmp(options.level,'group')
            if iscell(T0)
                auxj1 = cell2mat(T0(1:j-1));
                auxj = cell2mat(T0(1:j));
                t0 = sum(auxj1);
                t1 = sum(auxj);
                jj = [1:t0 (t1+1):sum(T)];
                Xj = X(jj,:); 
                Tj = [auxj1;  cell2mat(T0((j+1):end)) ];
                t0 = sum(auxj1) - length(auxj1)*hmm.train.maxorder;
                t1 = sum(auxj) - length(auxj)*hmm.train.maxorder;
                jj = [1:t0 (t1+1):size(Gamma,1) ];
                Gammaj = Gamma(jj,:);
            else
                t0 = sum(T(1:j-1)); jj = [1:t0 (sum(T(1:j))+1):sum(T)];
                Xj = X(jj,:); Tj = [T(1:j-1); T(j+1:end) ];
                t0 = sum(T(1:j-1)) - (j-1)*hmm.train.maxorder;
                t1 = sum(T(1:j)) - j*hmm.train.maxorder;
                jj = [1:t0 (t1+1):size(Gamma,1) ];
                Gammaj = Gamma(jj,:);
            end
        else % subject level estimation
            t0 = sum(T(1:j-1));
            Xj = X(t0+1:t0+T(j),:); Tj = T(j);
            t0 = sum(T(1:j-1)) - (j-1)*hmm.train.maxorder;
            Gammaj = Gamma(t0+1:t0+T(j)-hmm.train.maxorder,:);
            Gammasum(j,:) = sum(Gammaj);
        end
        %hmm0.train.zeromean = 0 ;
        hmm = mlhmmmar(Xj,Tj,hmm0,Gammaj,options.completelags);
    end
    
    for k=1:K
                
        setstateoptions;
        W = zeros(length(orders),ndim,ndim);
        for i=1:length(orders),
            %W(i,:,:) = loadings *  hmm.state(k).W.Mu_W(~zeromean + ((1:ndim) + (i-1)*ndim),:) * loadings' ;
            W(i,:,:) = hmm.state(k).W.Mu_W(~train.zeromean + ((1:ndim) + (i-1)*ndim),:);
        end
        %squeeze(W(:,:,1))
        
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
        
        for ff=1:options.Nf,
            A = zeros(ndim);
            for i=1:length(orders),
                o = orders(i);
                A = A + permute(W(i,:,:),[2 3 1]) * exp(-1i*w(ff)*o);
            end
            af_tmp = eye(ndim) - A;
            iaf_tmp = inv(af_tmp); % transfer matrix H
            psdc(ff,:,:,j,k) = iaf_tmp * covmk * iaf_tmp'; 
            ipsdc(ff,:,:,j,k) = af_tmp * preck * af_tmp';
                         
            % Get PDC
            if options.to_do(2)==1 && ndim>1
                for n=1:ndim,
                    for l=1:ndim,
                        pdcc(ff,n,l,j,k) = sqrt(preckd(n)) * abs(af_tmp(n,l)) / ...
                            sqrt( preckd * (abs(af_tmp(:,l)).^2) );
                    end
                end
            end
        end
        
        if strcmp(options.level,'subject') && options.to_do(1)==1 && ndim>1
            for n=1:ndim,
                for l=1:ndim,
                    rkj=psdc(:,n,l,j,k)./(sqrt(psdc(:,n,n,j,k)).*sqrt(psdc(:,l,l,j,k)));
                    cohc(:,n,l,j,k)=abs(rkj);
                    pcoh(:,n,l,j,k)=-ipsdc(:,n,l,j,k)./(sqrt(ipsdc(:,n,n,j,k)).*sqrt(ipsdc(:,l,l,j,k)));
                    phasec(:,n,l,j,k)=atan(imag(rkj)./real(rkj));
                end
            end
        end
        
    end
    
    %figure(4);clf(4);plot(abs(psdc(:,1,1,j,2)));pause(1)
    %j
    
end
    
for k=1:K
    
    fit.state(k).pdc = []; fit.state(k).coh = []; fit.state(k).pcoh = []; fit.state(k).phase = [];
    if strcmp(options.level,'group')
        fit.state(k).psd = mean(psdc(:,:,:,:,k),4);
        if options.to_do(2)==1  && ndim>1
            fit.state(k).pdc = mean(pdcc(:,:,:,:,k),4);
        end
    else
        fit.state(k).psdc = psdc(:,:,:,:,k);
        pGammasum = repmat(Gammasum(:,k),[1 ndim ndim options.Nf]);
        fit.state(k).psd = sum(psdc(:,:,:,:,k) .* permute(pGammasum,[4 2 3 1]),4) / sum(Gammasum(:,k));
        if options.to_do(2)==1  && ndim>1
            fit.state(k).pdc = sum(pdcc(:,:,:,:,k) .* permute(pGammasum,[4 2 3 1]),4) / sum(Gammasum(:,k));
        end
    end
    fit.state(k).ipsd = zeros(options.Nf,ndim,ndim);
    for ff=1:options.Nf,
        fit.state(k).ipsd(ff,:,:) = inv(permute(fit.state(k).psd(ff,:,:),[3 2 1]));
    end
    fit.state(k).f = freqs;
    
    % Get Coherence and Phase
    if options.to_do(1)==1 && ndim>1
        for n=1:ndim
            for l=1:ndim
                rkj=fit.state(k).psd(:,n,l)./(sqrt(fit.state(k).psd(:,n,n)).*sqrt(fit.state(k).psd(:,l,l)));
                fit.state(k).coh(:,n,l)=abs(rkj);
                fit.state(k).pcoh(:,n,l)=-fit.state(k).ipsd(:,n,l)./...
                    (sqrt(fit.state(k).ipsd(:,n,n)).*sqrt(fit.state(k).ipsd(:,l,l)));
                fit.state(k).phase(:,n,l)=atan(imag(rkj)./real(rkj));
            end
        end
        if strcmp(options.level,'subject')
            fit.state(k).cohc = cohc(:,:,:,:,k);
            fit.state(k).pcohc = pcohc(:,:,:,:,k);
            fit.state(k).phasec = phasec(:,:,:,:,k);
        end
    end
    
    if strcmp(options.level,'subject') && options.to_do(2)==1 && ndim>1
        fit.state(k).pdcc = pdcc(:,:,:,:,k);
    end
    
    if options.p>0 % jackknife
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
    fit.state(k).psd = (1/options.Fs) * fit.state(k).psd;
    % and take abs value for the diagonal
    for n=1:ndim, fit.state(k).psd(:,n,n) = abs(fit.state(k).psd(:,n,n)); end
    if options.p>0
        fit.state(k).psderr = (1/options.Fs) * fit.state(k).psderr;  
        for n=1:ndim, 
            fit.state(k).psderr(1,:,n,n) = abs(fit.state(k).psderr(1,:,n,n));
            fit.state(k).psderr(2,:,n,n) = abs(fit.state(k).psderr(2,:,n,n));
        end
    end
    if strcmp(options.level,'subject')
        fit.state(k).psdc = (1/options.Fs) * fit.state(k).psdc; 
        for n=1:ndim, fit.state(k).psdc(:,n,n,:) = abs(fit.state(k).psdc(:,n,n,:)); end
    end
end
end

