function fit = hmmspectrawt(data,T,Gamma,options)
%
% Computes wavelet power, coherence, phase and PDC, with intervals of
% confidence. Also obtains mean and standard deviation of the phases.
% PDC is approximated: From MT cross-spectral density, it obtains the transfer-function
% using the Wilson-Burg algorithm and then computes PDC.
% Intervals of confidence are computed using the jackknife over trials and tapers
%
%
% INPUTS:
% X: the data matrix, with all trials concatenated
% T: length of each trial
% Gamma: State time course  
% options       structure with the training options - see documentation in 
%                       https://github.com/OHBA-analysis/HMM-MAR/wiki
%               IMPORTANT: If the HMM was run with the options order or embeddedlags, 
%                           these must be specified here as well
%
%
% OUTPUTS:
%
% fit is a list with K elements, each of which contains:
% fit.state(k).psd: cross-spectral density (Nf x ndim x ndim)
% fit.state(k).ipsd inverse power spectral density matrix (Nf x ndim x ndim)
% fit.state(k).coh: coherence (Nf x ndim x ndim)
% fit.state(k).pcoh partial coherence (Nf x ndim x ndim)
% fit.state(k).pdc: partial directed coherence estimate (Nf x ndim x ndim)
% fit.state(k).phase: phase of the coherence, in degrees (Nf x ndim x ndim)
% fit.state(k).sdphase: jackknife standard deviation of the phase
% fit.state(k).f: frequencies (Nf x 1)
%       (where ndim is the number of channels) 
%
% Author: Cam Higgins and Diego Vidaurre, OHBA, University of Oxford (2014)
%  the code uses some parts from Chronux

if nargin<4, options = struct(); end
if ~isfield(options,'order') && ~isfield(options,'embeddedlags')
    warning(['If you ran the HMM with options.order>0 or options.embeddedlags, ' ... 
        'these need to be specified too here'])
end

if isfield(options,'Gamma'), options = rmfield(options,'Gamma'); end
if ~isfield(options,'order'), order = 0; options.order = 0;
else, order = options.order; 
end
if ~isfield(options,'embeddedlags'), embeddedlags = 0; 
else, embeddedlags = options.embeddedlags; 
end
if order>0 && length(embeddedlags)>1
    error('Order needs to be zero for multiple embedded lags')
end
if nargin<3 || isempty(Gamma)
    options.K = 1; 
else
    options.K = size(Gamma,2);
end
if ~isfield(options,'jointweights')
    options.jointweights = false; % this should be set to true for VRAD
end
% Adjust series lengths, and preprocess if data is not cell
if xor(iscell(data),iscell(T)), error('X and T must be cells, either both or none of them.'); end
if iscell(data)
    TT = []; 
    for j=1:length(data)
        t = double(T{j}); if size(t,1)==1, t = t'; end
        TT = [TT; t];
    end
    [options,~,ndim] = checkoptions_spectra(options,data,TT,0);
    if nargin<3 || isempty(Gamma)
        Gamma = ones(sum(TT),1);
    end
    if order > 0
        TT = TT - order; 
    elseif length(embeddedlags) > 1
        d1 = -min(0,embeddedlags(1));
        d2 = max(0,embeddedlags(end));
        TT = TT - (d1+d2);
    end
else
    T = double(T);
    [options,~,ndim] = checkoptions_spectra(options,data,T,0);
    if nargin<3 || isempty(Gamma)
        Gamma = ones(sum(T),1); options.order = 0; options.embeddedlags = 0;
    end 
    % Standardise data and control for ackward trials
    if options.standardise
        data = standardisedata(data,T,options.standardise);
    end
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
    % remove the exceeding part of X (with no attached Gamma)
    if order > 0
        data2 = zeros(sum(T)-length(T)*order,ndim);
        for n = 1:length(T)
            t0 = sum(T(1:n-1)); t00 = sum(T(1:n-1)) - (n-1)*order;
            data2(t00+1:t00+T(n)-order,:) = data(t0+1+order:t0+T(n),:);
        end
        T = T - order;
        data = data2; clear data2;
    elseif length(embeddedlags) > 1
        d1 = -min(0,embeddedlags(1));
        d2 = max(0,embeddedlags(end));
        data2 = zeros(sum(T)-length(T)*(d1+d2),ndim);
        for n = 1:length(T)
            idx1 = sum(T(1:n-1))-(d1+d2)*(n-1)+1; 
            idx2 = sum(T(1:n)) - (d1+d2)*n; 
            idx3 = sum(T(1:n-1))+d1+1; 
            idx4 = sum(T(1:n)) - d2 ;
            data2(idx1:idx2,:) = data(idx3:idx4,:); 
        end
        T = T - (d1+d2);
        data = data2; clear data2;  
    end
    TT = T;
end

K = size(Gamma,2);

if options.p>0, options.err = [2 options.p]; end
if isfield(options,'downsample') && options.downsample~=0
    Fs = options.downsample;
else
    Fs = options.Fs;
end
fit = {};

% set wavelet params:

S = [];
S.tf_method = 'morlet';
S.tf_freq_range = options.fpass;
Nf = 2*round(diff(options.fpass));
S.tf_num_freqs = Nf; % default to approx 0.5Hz resolution
%S.raw_times = [1:length(datap)]./Hz;
S.ds_factor =1;
S.tf_morlet_factor = 3;
%S.tf_morlet_basis = morletbasisset;
S.tf_calc_amplitude = false; % return real and complex parts

f = linspace(options.fpass(1),options.fpass(2),S.tf_num_freqs);
% nfft = max(2^(nextpow2(options.win)+options.pad),options.win);
% [f,findx]=getfgrid(Fs,nfft,options.fpass);
% Nf = length(f); options.Nf = Nf;
% tapers=dpsschk(options.tapers,options.win,Fs); % check tapers
% ntapers = options.tapers(2);

if ~isfield(options,'hammingweighted');
    options.hammingweighted = false;
end
           
% Wavelet Cross-frequency matrix calculation
if options.p > 0
    error('Jackknife confidence intervals not implemented for wavelete estimation');
    psdc = zeros(Nf,ndim,ndim,length(TT),ntapers);
else
    psdc = zeros(Nf,ndim,ndim);
end
sumgamma = zeros(K,1);
stime = zeros(length(TT),1);
c = 1;  
t00 = 0; 
NwinsALL = 0;
psdc = zeros(K,Nf,ndim,ndim);
for n = 1:length(T)
    if iscell(data)
        X = loadfile(data{n},T{n},options); % includes preprocessing
        if order > 0
            X2 = zeros(sum(T{n})-length(T{n})*order,ndim);
            for nn = 1:length(T{n})
                t0_star = sum(T{n}(1:nn-1)); t00_star = sum(T{n}(1:nn-1)) - (nn-1)*order;
                X2(t00_star+1:t00_star+T{n}(nn)-order,:) = X(t0_star+1+order:t0_star+T{n}(nn),:);
            end
            X = X2; clear X2;
        elseif length(embeddedlags) > 1                
            d1 = -min(0,embeddedlags(1));
            d2 = max(0,embeddedlags(end));
            X2 = zeros(sum(T{n})-length(T{n})*(d1+d2),ndim);
            for nn = 1:length(T{n})
                idx1 = sum(T{n}(1:nn-1))-(d1+d2)*(nn-1)+1;
                idx2 = sum(T{n}(1:nn)) - (d1+d2)*nn;
                idx3 = sum(T{n}(1:nn-1))+d1+1;
                idx4 = sum(T{n}(1:nn)) - d2 ;
                X2(idx1:idx2,:) = X(idx3:idx4,:);
            end
            X = X2; clear X2;
        end
        LT = length(T{n});
    else
        X = data((1:TT(n)) + sum(TT(1:n-1)) , : ); 
        LT= 1;
    end
    t0 = 0;
    
    for nn = 1:LT % elements for this subject
        ind_gamma = (1:TT(c)) + t00;
        ind_X = (1:TT(c)) + t0;
        t00 = t00 + TT(c); t0 = t0 + TT(c);
        
%             Nwins = round(TT(c)/options.win); % pieces are going to be included as windows only if long enough
%             NwinsALL = NwinsALL + Nwins; 
        S.raw_times = [1:length(X)]./Fs;
        tempdatatf = osl_tf_transform(S,X');
        
        if ~all(tempdatatf.tf_freqs==f)
            error('Something wrong: frequency resolution mismatch');
        end
        if options.jointweights
            for ifreq = 1:length(f)
                %temp1 = tempdatatf.dattf(:,:,1,ifreq)
                temp = zeros(TT(c),ndim,ndim);
                for t=1:TT(c)
                    temp(t,:,:) = tempdatatf.dattf(:,t,1,ifreq)*tempdatatf.dattf(:,t,1,ifreq)';
                end
                %regress out all state effects in one go:
                if options.hammingweighted
                    weights = zeros(TT(c),K);
                    for k=1:K
                        weights(:,k) = conv(Gamma(ind_gamma,:),abs(tempdatatf.tf_morlet_basis{ifreq}),'same');
                    end
                    weights = weights./max(weights);
                else
                    weights = Gamma(ind_gamma,:);
                end
                psdtemp = pinv(weights)*temp(:,:);
                psdc(:,ifreq,:,:) = reshape(psdtemp,[K,1,ndim,ndim]);
            end
            clear psdtemp temp
        else
            for k=1:K
                try
                    if sum(Gamma(ind_gamma,k))<10,
                        psdc(k,:,:,:) = 0;
                        continue; 
                    end
                    
                catch
                    error('Index exceeds matrix dimensions - Did you specify options correctly? ')
                end
                for ifreq = 1:length(f)
                    temp1 = repmat(Gamma(ind_gamma,k)',ndim,1).*tempdatatf.dattf(:,:,1,ifreq);
                    temp = temp1 * tempdatatf.dattf(:,:,1,ifreq)';
                    psdc(k,ifreq,:,:) = squeeze(psdc(k,ifreq,:,:)) + temp;
                end
                sumgamma(k) = sumgamma(k) + sum(Gamma(ind_gamma,k));
            end
        end
%             for iwin = 1:Nwins
%                 ranget = (iwin-1)*options.win+1:iwin*options.win;
%                 if ranget(end)>TT(c)
%                     nzeros = ranget(end) - TT(c);
%                     nzeros2 = floor(double(nzeros)/2) * ones(2,1);
%                     if sum(nzeros2)<nzeros, nzeros2(1) = nzeros2(1)+1; end
%                     ranget = ranget(1):TT(c);
%                     Xwin = [zeros(nzeros2(1),ndim); Xki(ranget,:); zeros(nzeros2(2),ndim)]; % padding with zeroes
%                 else
%                     Xwin = Xki(ranget,:);
%                 end 
%                 J = mtfftc(Xwin,tapers,nfft,Fs); 
%                 sumgamma(c) = sumgamma(c) + sum(Gamma(ind_gamma(ranget),k).^2); 
%                 stime(c) = stime(c) + length(Gamma(ind_gamma(ranget),k)); 
%                 for tp = 1:ntapers              
%                     Jik = J(findx,tp,:);
%                     for j = 1:ndim
%                         for l = j:ndim
%                             if options.p > 0
%                                 psdc(:,j,l,c,tp) = psdc(:,j,l,c,tp) + conj(Jik(:,1,j)).*Jik(:,1,l); 
%                                 if l~=j
%                                     psdc(:,l,j,(c-1)*ntapers+tp) = conj(psdc(:,j,l,(c-1)*ntapers+tp));
%                                 end
%                             else
%                                 psdc(:,j,l) = psdc(:,j,l) + conj(Jik(:,1,j)).*Jik(:,1,l);  
%                                 if l~=j
%                                     psdc(:,l,j) = conj(psdc(:,j,l));
%                                 end
%                             end
%                         end
%                     end
%                 end
%             end
        
        
        c = c + 1;
    end
    
    if options.verbose, disp(['Segment ' num2str(n) ', state ' num2str(k)]); end
end

%     if options.p > 0
%         sumgamma = sumgamma(1:c-1); 
%         stime = stime(1:c-1); 
%         psdc = psdc(:,:,:,1:c-1,:);
%         psd = sum(sum(psdc,5),4) / (sum(sumgamma) / sum(stime)) / ntapers / NwinsALL;
%         for iNf = 1:Nf
%             for tp = 1:ntapers
%                 for indim=1:ndim
%                     for indim2=1:ndim
%                         psdc(iNf,indim,indim2,:,tp) = permute(psdc(iNf,indim,indim2,:,tp),[4 1 2 3]) ...
%                             ./ (sumgamma ./ stime) ;
%                     end
%                 end
%             end
%         end
%         psdc = reshape(psdc,[Nf,ndim,ndim,size(psdc,4)*ntapers]);
%     else

%     end
for k=1:K
    psdc(k,:,:,:) = psdc(k,:,:,:) / sum(sumgamma(k));
    
    psd = squeeze(psdc(k,:,:,:));
    ipsd = zeros(Nf,ndim,ndim);
    
    for ff = 1:Nf
        if rcond(permute(psd(ff,:,:),[3 2 1]))>1e-10
            ipsd(ff,:,:) = pinv(permute(psd(ff,:,:),[3 2 1])); 
        else
            ipsd(ff,:,:) = nan;
        end
    end
    
    % coherence
    coh = []; pcoh = []; phase = []; pdc = [];
    if (options.to_do(1)==1) && ndim>1
        coh = zeros(Nf,ndim,ndim); phase = zeros(Nf,ndim,ndim);
        for j = 1:ndim
            for l = 1:ndim
                cjl = psd(:,j,l)./sqrt(psd(:,j,j) .* psd(:,l,l));
                coh(:,j,l) = abs(cjl); 
                cjlp = -ipsd(:,j,l)./sqrt(ipsd(:,j,j) .* ipsd(:,l,l));
                pcoh(:,j,l) = abs(cjlp);
                phase(:,j,l) = angle(cjl); %atan(imag(rkj)./real(rkj));
            end
        end
    end 
    
    if (options.to_do(2)==1) && ndim>1
        [pdc, dtf] = subrutpdc(psd,options.numIterations,options.tol);
    end
    
    if options.p>0 % jackknife
        disp('Jackknifing now... (this might take time - set options.p==0 to skip)'); 
        [psderr,coherr,pcoherr,pdcerr,sdphase] = spectrerr(psdc,[],coh,pcoh,pdc,options);
    end
    
    if options.rlowess
        for j=1:ndim
            psd(:,j,j) = mslowess(f', psd(:,j,j));
            if options.p>0
                psderr(1,:,j,j) = mslowess(f', squeeze(psderr(1,:,j,j))');
                psderr(2,:,j,j) = mslowess(f', squeeze(psderr(2,:,j,j))');
            end
            for l=1:ndim
                if (options.to_do(1)==1)
                    coh(:,j,l) = mslowess(f', coh(:,j,l));
                    pcoh(:,j,l) = mslowess(f', pcoh(:,j,l));
                end
                if (options.to_do(2)==1)
                    pdc(:,j,l) = mslowess(f', pdc(:,j,l));
                end
                if options.p>0
                    if (options.to_do(1)==1)
                        coherr(1,:,j,l) = mslowess(f', squeeze(coherr(1,:,j,l))');
                        coherr(2,:,j,l) = mslowess(f', squeeze(coherr(2,:,j,l))');
                        pcoherr(1,:,j,l) = mslowess(f', squeeze(pcoherr(1,:,j,l))');
                        pcoherr(2,:,j,l) = mslowess(f', squeeze(pcoherr(2,:,j,l))');
                    end
                    if (options.to_do(2)==1)
                        pdcerr(1,:,j,l) = mslowess(f', squeeze(pdcerr(1,:,j,l))');
                        pdcerr(2,:,j,l) = mslowess(f', squeeze(pdcerr(2,:,j,l))');
                    end
                end
            end
        end
    end
    
    fit.state(k).f = f;
    fit.state(k).psd = psd;
    fit.state(k).ipsd = ipsd;
    if (options.to_do(1)==1) && ndim>1
        fit.state(k).coh = coh;
        fit.state(k).pcoh = pcoh;
        fit.state(k).phase = phase;
    end
    if (options.to_do(2)==1) && ndim>1
        fit.state(k).pdc = pdc;
        fit.state(k).dtf = dtf;
    end
    if options.p>0
        fit.state(k).psderr = psderr;
        if (options.to_do(1)==1) && ndim>1
            fit.state(k).coherr = coherr;
            fit.state(k).pcoherr = pcoherr;
            fit.state(k).sdphase = sdphase;
        end
        if (options.to_do(2)==1) && ndim>1
            fit.state(k).pdcerr = pdcerr;
        end
    end
%end

%fit.state(k).ntpts=size(X,1);

end

