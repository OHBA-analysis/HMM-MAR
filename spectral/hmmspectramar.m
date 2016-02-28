function fit = hmmspectramar(X,T,hmm,Gamma,options)
% Get ML spectral estimates from MAR model
%
% INPUT
% X             time series 
% T             Number of time points for each time series
% hmm           An hmm-mar structure 
% Gamma         State time course (not used if options.MLestimation=0)
% options 

%  .Fs:       Sampling frequency
%  .fpass:    Frequency band to be used [fmin fmax] (default [0 fs/2])
%  .p:        p-value for computing jackknife confidence intervals (default 0)
%  .Nf        No. of frequencies to be computed in the range minHz-maxHz
%  .MLestimation     State responsabilities - if 0, it will use the
%               MAR models as they were returned by the HMM-MAR inference (i.e. a
%               posterior distribution instead of maximum likelihood)
%  .completelags    if MLestimation is true and completelags is true,
%               the MAR spectra will be calculated with the complete set of
%               lags up to the specified order (lags=1,2,...,order)
%  .order     If we want a higher MAR order than that used for training,
%               specify it here - a new set of MAR models will be estimated
%
% OUTPUT
% fit is a list with K elements, each of which contains: 
% fit.state(k).psd     [Nf x N x N] Power Spectral Density matrix
% fit.state(k).ipsd     [Nf x N x N] Inverse Power Spectral Density matrix
% fit.state(k).coh     [Nf x N x N] Coherence matrix
% fit.state(k).pcoh    [Nf x N x N] Partial Coherence matrix
% fit.state(k).pdc   [Nf x N x N] Baccala's Partial Directed Coherence
% fit.state(k).phase     [Nf x N x N] Phase matrix
% fit.state(k).psderr: interval of confidence for the cross-spectral density (2 x Nf x N x N)
% fit.state(k).coherr: interval of confidence for the coherence (2 x Nf x N x N)
% fit.state(k).pcoherr: interval of confidence for the partial coherence (2 x Nf x N x N)
% fit.state(k).pdcerr: interval of confidence for the partial directed coherence (2 x Nf x N x N)
% fit.state(k).f     [Nf x 1] Frequency vector
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2014)

sT = sum(T);
ndim = size(X,2);

if ~isempty(hmm), K = length(hmm.state); 
else K = size(Gamma,2);
end

if isfield(options,'order') && ~isempty(hmm)
    for k=1:K, 
        hmm.state(k).order = options.order; 
        if isfield(hmm.state(k),'train'), hmm.state(k).train.order = options.order; end
    end
    hmm.train.order = options.order;
end

if isfield(options,'order') && options.order~=hmm.train.maxorder
   if options.order<hmm.train.maxorder
       error('If you specify a new MAR order, has to be higher than hmm.train.maxorder')
   end
   Gamma2 = zeros(sT-length(T)*options.order,K);
   for in = 1:length(T),
       t0 = sum(T(1:in-1)) - (in-1)*hmm.train.order;
       t00 = sum(T(1:in-1)) - (in-1)*options.order;
       Gamma2(t00+1:t00+T(in)-options.order,:) = ...
           Gamma(t0+1+options.order:t0+T(in),:);
   end
   Gamma = Gamma2; clear Gamma2
   hmm.train.maxorder = options.order; 
end

options = checkoptions_spectra(options,ndim,T);

if hmm.train.maxorder==0
    error('MAR spectra cannot be estimated for MAR order equal to 0')
end

if length(T)<5 && options.p>0,  
    error('You need at least 5 trials to compute error bars for MAR spectra'); 
end

%loadings = options.loadings;
%if hmm.train.whitening, loadings = loadings * iA; end
%M = size(options.loadings,1);

freqs = (0:options.Nf-1)* ...
    ( (options.fpass(2) - options.fpass(1)) / (options.Nf-1)) + options.fpass(1);
w=2*pi*freqs/options.Fs;

if options.p==0, N = 1; 
else N = length(T); end

if options.p==0
    psdc = zeros(options.Nf,ndim,ndim,N,K);
    pdcc = zeros(options.Nf,ndim,ndim,N,K);
    ipsdc = zeros(options.Nf,ndim,ndim,N,K);
    % cohc = zeros(options.Nf,ndim,ndim,N,K);
    % pcohc = zeros(options.Nf,ndim,ndim,N,K);
    % phasec = zeros(options.Nf,ndim,ndim,N,K);
    NN = 1; 
else
    psdc = zeros(options.Nf,ndim,ndim,1,K);
    pdcc = zeros(options.Nf,ndim,ndim,1,K);
    ipsdc = zeros(options.Nf,ndim,ndim,1,K);
    % cohc = zeros(options.Nf,ndim,ndim,1,K);
    % pcohc = zeros(options.Nf,ndim,ndim,1,K);
    % phasec = zeros(options.Nf,ndim,ndim,1,K);    
    NN = N;
end

if size(T,1)==1, T=T'; end
hmm0 = hmm;

for j=1:NN
    
    if options.p==0
        Gammaj = Gamma; Xj = X; Tj = T;
    else
        t0 = sum(T(1:j-1)); jj = [1:t0 (sum(T(1:j))+1):sT]; 
        Xj = X(jj,:); Tj = [T(1:j-1); T(j+1:end) ]; 
        t0 = sum(T(1:j-1)) - (j-1)*hmm.train.maxorder; 
        jj = [1:t0 (sum(T(1:j))-j*hmm.train.maxorder+1):(sT-length(T)*hmm.train.maxorder)]; 
        Gammaj = Gamma(jj,:);
    end
    
    if options.MLestimation
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
        
        switch train.covtype
            case 'uniquediag'
                covmk = diag(hmm.Omega.Gam_rate / hmm.Omega.Gam_shape);
                preck = diag(hmm.Omega.Gam_shape ./ hmm.Omega.Gam_rate);
            case 'diag'
                covmk = diag(hmm.state(k).Omega.Gam_rate / hmm.state(k).Omega.Gam_shape);
                preck = diag(hmm.state(k).Omega.Gam_shape ./ hmm.state(k).Omega.Gam_rate);
            case 'uniquefull'
                covmk = hmm.Omega.Gam_rate ./ hmm.Omega.Gam_shape;
                preck = inv(covmk);
            case 'full'
                covmk = hmm.state(k).Omega.Gam_rate ./ hmm.state(k).Omega.Gam_shape;
                preck = inv(covmk);
        end
        
        % Get Power Spectral Density matrix and PDC for state K
        
        for ff=1:options.Nf,
            af_tmp=eye(ndim);
            for i=1:length(orders),
                o = orders(i);
                af_tmp=af_tmp - permute(W(i,:,:),[2 3 1]) * exp(-1i*w(ff)*o);
            end
            iaf_tmp=inv(af_tmp);
            psdc(ff,:,:,j,k) = iaf_tmp * covmk * iaf_tmp';
            ipsdc(ff,:,:,j,k) = inv(permute(psdc(ff,:,:,j,k),[2 3 1]));
                         
            % Get PDC
            if options.to_do(2)==1
                for n=1:ndim,
                    prec_nn=1/sqrt(covmk(n,n));
                    for l=1:ndim,
                        pdcc(ff,n,l,j,k) = prec_nn * abs(af_tmp(n,l))/sqrt(abs(af_tmp(:,l)'*preck*af_tmp(:,l)));
                    end
                end
            end
        end
        
%         for n=1:ndim,
%             for l=1:ndim,
%                 rkj=psdc(:,n,l,j,k)./(sqrt(psdc(:,n,n,j,k)).*sqrt(psdc(:,l,l,j,k)));
%                 cohc(:,n,l,j,k)=abs(rkj);
%                 pcoh(:,n,l,j,k)=-ipsdc(:,n,l,j,k)./(sqrt(ipsdc(:,n,n,j,k)).*sqrt(ipsdc(:,l,l,j,k)));
%                 phasec(:,n,l,j,k)=atan(imag(rkj)./real(rkj));
%             end
%         end
        
    end
    
end
    
for k=1:K
    
    fit.state(k).pdc = []; fit.state(k).coh = []; fit.state(k).pcoh = []; fit.state(k).phase = [];
    fit.state(k).psd = mean(psdc(:,:,:,:,k),4);
    fit.state(k).ipsd = mean(ipsdc(:,:,:,:,k),4);
    if options.to_do(2)==1, 
        fit.state(k).pdc = mean(pdcc(:,:,:,:,k),4);
    end
    for ff=1:options.Nf, fit.state(k).ipsd(ff,:,:) = inv(permute(fit.state(k).psd(ff,:,:),[3 2 1])); end    
    fit.state(k).f = freqs;
    
    % Get Coherence and Phase
    if options.to_do(1)==1
        for n=1:ndim,
            for l=1:ndim,
                rkj=fit.state(k).psd(:,n,l)./(sqrt(fit.state(k).psd(:,n,n)).*sqrt(fit.state(k).psd(:,l,l)));
                fit.state(k).coh(:,n,l)=abs(rkj);
                fit.state(k).pcoh(:,n,l)=-fit.state(k).ipsd(:,n,l)./(sqrt(fit.state(k).ipsd(:,n,n)).*sqrt(fit.state(k).ipsd(:,l,l)));
                fit.state(k).phase(:,n,l)=atan(imag(rkj)./real(rkj));
            end
        end
    end
    
    if options.p>0 % jackknife
        [psderr,coherr,pcoherr,pdcerr,sdphase] = spectrerr(psdc(:,:,:,:,k),pdcc(:,:,:,:,k),fit.state(k).coh, ...
            fit.state(k).pcoh,fit.state(k).pdc,options,1);
        fit.state(k).psderr = psderr;
        if options.to_do(1)==1
            fit.state(k).coherr = coherr;
            fit.state(k).pcoherr = pcoherr;
            fit.state(k).sdphase = sdphase;
        end
        if options.to_do(2)==1
            fit.state(k).pdcerr = pdcerr;
        end
    end
    
    % weight the PSD by the inverse of the sampling rate
    fit.state(k).psd = (1/options.Fs) * fit.state(k).psd;  
    for n=1:ndim, fit.state(k).psd(:,n,n) = abs(fit.state(k).psd(:,n,n)); end
    if options.p>0
        fit.state(k).psderr = (1/options.Fs) * fit.state(k).psderr;  
        for n=1:ndim, 
            fit.state(k).psderr(1,:,n,n) = abs(fit.state(k).psderr(1,:,n,n));
            fit.state(k).psderr(2,:,n,n) = abs(fit.state(k).psderr(2,:,n,n));
        end
    end
end
end

