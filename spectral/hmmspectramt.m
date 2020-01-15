function fit = hmmspectramt(data,T,Gamma,options)
%
% Computes nonparametric (multitaper) power, coherence, phase and PDC, with intervals of
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
% fit.state(k).psderr: interval of confidence for the cross-spectral density (2 x Nf x ndim x ndim)
% fit.state(k).coherr: interval of confidence for the coherence (2 x Nf x ndim x ndim)
% fit.state(k).pcoherr: interval of confidence for the partial coherence (2 x Nf x ndim x ndim)
% fit.state(k).pdcerr: interval of confidence for the partial directed coherence (2 x Nf x ndim x ndim)
% fit.state(k).sdphase: jackknife standard deviation of the phase
% fit.state(k).f: frequencies (Nf x 1)
%       (where ndim is the number of channels) 
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2014)
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

nfft = max(2^(nextpow2(options.win)+options.pad),options.win);
[f,findx]=getfgrid(Fs,nfft,options.fpass);
Nf = length(f); options.Nf = Nf;
tapers=dpsschk(options.tapers,options.win,Fs); % check tapers
ntapers = options.tapers(2);

for k = 1:K
           
    % Multitaper Cross-frequency matrix calculation
    if options.p > 0
        psdc = zeros(Nf,ndim,ndim,length(TT),ntapers);
    else
        psdc = zeros(Nf,ndim,ndim);
    end
    sumgamma = zeros(length(TT),1);
    stime = zeros(length(TT),1);
    c = 1;  
    t00 = 0; 
    NwinsALL = 0;
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
            try
                if sum(Gamma(ind_gamma,k))<10, c = c + 1; continue; end
                Xki = X(ind_X,:) .* repmat(Gamma(ind_gamma,k),1,ndim);
            catch
                error('Index exceeds matrix dimensions - Did you specify options correctly? ')
            end
            Nwins = round(TT(c)/options.win); % pieces are going to be included as windows only if long enough
            NwinsALL = NwinsALL + Nwins; 
            
            for iwin = 1:Nwins
                ranget = (iwin-1)*options.win+1:iwin*options.win;
                if ranget(end)>TT(c)
                    nzeros = ranget(end) - TT(c);
                    nzeros2 = floor(double(nzeros)/2) * ones(2,1);
                    if sum(nzeros2)<nzeros, nzeros2(1) = nzeros2(1)+1; end
                    ranget = ranget(1):TT(c);
                    Xwin = [zeros(nzeros2(1),ndim); Xki(ranget,:); zeros(nzeros2(2),ndim)]; % padding with zeroes
                else
                    Xwin = Xki(ranget,:);
                end 
                J = mtfftc(Xwin,tapers,nfft,Fs); 
                sumgamma(c) = sumgamma(c) + sum(Gamma(ind_gamma(ranget),k).^2); 
                stime(c) = stime(c) + length(Gamma(ind_gamma(ranget),k)); 
                for tp = 1:ntapers              
                    Jik = J(findx,tp,:);
                    for j = 1:ndim
                        for l = j:ndim
                            if options.p > 0
                                psdc(:,j,l,c,tp) = psdc(:,j,l,c,tp) + conj(Jik(:,1,j)).*Jik(:,1,l); 
                                if l~=j
                                    psdc(:,l,j,(c-1)*ntapers+tp) = conj(psdc(:,j,l,(c-1)*ntapers+tp));
                                end
                            else
                                psdc(:,j,l) = psdc(:,j,l) + conj(Jik(:,1,j)).*Jik(:,1,l);  
                                if l~=j
                                    psdc(:,l,j) = conj(psdc(:,j,l));
                                end
                            end
                        end
                    end
                end
            end
            c = c + 1;
        end
        if options.verbose, disp(['Segment ' num2str(n) ', state ' num2str(k)]); end
    end
    if options.p > 0
        sumgamma = sumgamma(1:c-1); 
        stime = stime(1:c-1); 
        psdc = psdc(:,:,:,1:c-1,:);
        psd = sum(sum(psdc,5),4) / (sum(sumgamma) / sum(stime)) / ntapers / NwinsALL;
        for iNf = 1:Nf
            for tp = 1:ntapers
                for indim=1:ndim
                    for indim2=1:ndim
                        psdc(iNf,indim,indim2,:,tp) = permute(psdc(iNf,indim,indim2,:,tp),[4 1 2 3]) ...
                            ./ (sumgamma ./ stime) ;
                    end
                end
            end
        end
        psdc = reshape(psdc,[Nf,ndim,ndim,size(psdc,4)*ntapers]);
    else
        psd = psdc / (sum(sumgamma) / sum(stime)) / ntapers / NwinsALL;
    end
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
end

%fit.state(k).ntpts=size(X,1);

end



function [tapers,eigs]=dpsschk(tapers,N,Fs)
% calculates tapers and, if precalculated tapers are supplied,
% checks that they (the precalculated tapers) the same length in time as
% the time series being studied. The length of the time series is specified
% as the second input argument N. Thus if precalculated tapers have
% dimensions [N1 K], we require that N1=N.
%
% From Chronux

if nargin < 3; error('Need all arguments'); end
sz=size(tapers);
if isempty(strfind(which('dpss'),matlabroot))
    error(['Function dpss() seems to be other than Matlab''s own ' ...
        '- you need to rmpath() it. Use ''rmpath(fileparts(which(''dpss'')))'''])
end

if sz(1)==1 && sz(2)==2
    try
        [tapers,eigs]=dpss(N,tapers(1),tapers(2));
    catch 
        error('Window is too short - increase options.win')
    end
    tapers = tapers*sqrt(Fs);
elseif N~=sz(1)
    error(['error in your dpss calculation? ' ...
        'the number of time points is different from the length of the tapers']);
end
end

%-------------------------------------------------------------------

function [f,findx]=getfgrid(Fs,nfft,fpass)
% gets the frequency grid associated with a given fft based computation
% Called by spectral estimation routines to generate the frequency axes
%
% From Chronux

if nargin < 3; error('Need all arguments'); end;
df=Fs/nfft;
f=0:df:Fs; % all possible frequencies
f=f(1:nfft);
if length(fpass)~=1
    findx=find(f>=fpass(1) & f<=fpass(end));
else
    [~,findx]=min(abs(f-fpass));
end
f=f(findx);
end


function J = mtfftc(data,tapers,nfft,Fs)
% Multi-taper fourier transform - continuous data
%
% Usage:
% J=mtfftc(data,tapers,nfft,Fs) - all arguments required
% Input:
%       data (in form samples x channels/trials or a single vector)
%       tapers (precalculated tapers from dpss)
%       nfft (length of padded data)
%       Fs   (sampling frequency)
%
% Output:
%       J (fft in form frequency index x taper index x channels/trials)
%
% From Chronux

if nargin < 4; error('Need all input arguments'); end
dtmp=[];
if isstruct(data)
    C=length(data);
    if C==1
        fnames=fieldnames(data);
        eval(['dtmp=data.' fnames{1} ';'])
        data=dtmp(:);
    end
else
    [N,C]=size(data);
    if N==1 || C==1
        data=data(:);
    end
end
[NC,C]=size(data); % size of data
[NK, K]=size(tapers); % size of tapers
if NK~=NC; error('length of tapers is incompatible with length of data'); end
tapers=tapers(:,:,ones(1,C)); % add channel indices to tapers
data=data(:,:,ones(1,K)); % add taper indices to data
data=permute(data,[1 3 2]); % reshape data to get dimensions to match those of tapers
data_proj=data.*tapers; % product of data with tapers
J=fft(data_proj,nfft)/Fs;   % fft of projected data
end

% function X = loadfile_mt(f,T,options)
% 
% if ischar(f)
%     fsub = f;
%     loadfile_sub;
% else
%     X = f;
% end
% if options.standardise == 1
%     for i=1:length(T)
%         t = (1:T(i)) + sum(T(1:i-1));
%         X(t,:) = X(t,:) - repmat(mean(X(t,:)),length(t),1);
%         X(t,:) = X(t,:) ./ repmat(std(X(t,:)),length(t),1);
%     end
% end
% end