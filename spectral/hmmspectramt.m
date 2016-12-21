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
% Gamma: State time course (not used if options.MLestimation=0)
% options: include the following fields
%   .Gamma: Estimated posterior probabilities of the states (default: all ones)
%   .tapers: A numeric vector [TW K] where TW is the
%       time-bandwidth product and K is the number of
%       tapers to be used (less than or equal to
%       2TW-1)
%   .win: number of time points per non-overlapping window
%   .Fs: Sampling frequency
%   .fpass: Frequency band to be used [fmin fmax] (default [0 fs/2])
%   .p: p-value for computing jackknife confidence intervals (default 0)
%   .to_do: a (2 by 1) vector, with component indicating, respectively, 
% 		whether a estimation of coherence and/or PDC is going to be produced (default is [1 1])
%   .numIterations: no. iterations for the Wilson algorithm (default: 100)
%   .tol: tolerance limit (default: 1e-18)
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

% if iscell(T)
%     for i = 1:length(T)
%         if size(T{i},1)==1, T{i} = T{i}'; end
%     end
%     T = cell2mat(T);
% end

if iscell(data)
    if ~isfield(options,'standardise'), options.standardise = 0; end
    X = loadfile_mt(data{1},T{1},options);
    ndim = size(X,2);
    TT = [];
    for j=1:length(data)
        t = double(T{j}); if size(t,1)==1, t = t'; end
        X = loadfile_mt(data{1},T{1},options);
        TT = [TT; t];
    end
    options = checkoptions_spectra(options,ndim,TT);
    if nargin<3 || isempty('Gamma')
        Gamma = ones(sum(TT),1);
    end
    order = (sum(TT) - size(Gamma,1)) / length(TT);
    TT = TT - order;
else
    ndim = size(data,2); T = double(T); 
    options = checkoptions_spectra(options,ndim,T);
    if nargin<3 || ~isempty('Gamma'),
        Gamma = ones(sum(T),1);
    end
    order = (sum(T) - size(Gamma,1)) / length(T);
    % remove the exceeding part of X (with no attached Gamma)
    if order>0
        data2 = zeros(sum(T)-length(T)*order,ndim);
        for in = 1:length(T),
            t0 = sum(T(1:in-1)); t00 = sum(T(1:in-1)) - (in-1)*order;
            data2(t00+1:t00+T(in)-order,:) = data(t0+1+order:t0+T(in),:);
        end
        T = T - order;
        data = data2; clear data2;
    end
    TT = T;
end

Gamma = sqrt(Gamma) .* repmat( sqrt(size(Gamma,1) ./ sum(Gamma)), size(Gamma,1), 1);

K = size(Gamma,2);

if options.p>0, options.err = [2 options.p]; end
Fs = options.Fs;
fit = {};

nfft = max(2^(nextpow2(options.win)+options.pad),options.win);
[f,findx]=getfgrid(Fs,nfft,options.fpass);
Nf = length(f); options.Nf = Nf;
tapers=dpsschk(options.tapers,options.win,Fs); % check tapers
ntapers = options.tapers(2);


for k=1:K
           
    % Multitaper Cross-frequency matrix calculation
    psdc = zeros(Nf,ndim,ndim,length(TT)*ntapers);
    sumgamma = zeros(length(TT)*ntapers,1);
    c = 1;  
    t00 = 0; 
    for in=1:length(T)
        if iscell(data)
            X = loadfile_mt(data{in},T{in},options);
            if order>0
                X2 = zeros(sum(T{in})-length(T{in})*order,ndim);
                for inn = 1:length(T{in}),
                    t0_star = sum(T{in}(1:inn-1)); t00_star = sum(T{in}(1:inn-1)) - (inn-1)*order;
                    X2(t00_star+1:t00_star+T{in}(inn)-order,:) = X(t0_star+1+order:t0_star+T{in}(inn),:);
                end
                X = X2; clear X2;
            end
            LT = length(T{in});
        else
            X = data((1:TT(in)) + sum(TT(1:in-1)) , : ); 
            LT= 1;
        end
        t0 = 0;
        for inn=1:LT
            ind_gamma = (1:TT(c)) + t00;
            ind_X = (1:TT(c)) + t0;
            t00 = t00 + TT(c); t0 = t0 + TT(c);
            if sum(Gamma(ind_gamma,k))<1, c = c + 1; continue; end
            Xki = X(ind_X,:) .* repmat(Gamma(ind_gamma,k),1,ndim);
            Nwins=round(TT(c)/options.win); % pieces are going to be included as windows only if long enough
            for iwin=1:Nwins,
                ranget = (iwin-1)*options.win+1:iwin*options.win;
                if ranget(end)>TT(c),
                    nzeros = ranget(end) - TT(c);
                    nzeros2 = floor(double(nzeros)/2) * ones(2,1);
                    if sum(nzeros2)<nzeros, nzeros2(1) = nzeros2(1)+1; end
                    ranget = ranget(1):TT(c);
                    Xwin=[zeros(nzeros2(1),ndim); Xki(ranget,:); zeros(nzeros2(2),ndim)]; % padding with zeroes
                    
                else
                    Xwin=Xki(ranget,:);
                end
                J=mtfftc(Xwin,tapers,nfft,Fs); % use detrend on X?
                sumgamma((c-1)*ntapers+(1:ntapers)) = sumgamma((c-1)*ntapers+(1:ntapers)) + sum(Gamma(ind_gamma(ranget),k)); 
                for tp=1:ntapers,
                    Jik=J(findx,tp,:);
                    for j=1:ndim
                        %for l=1:ndim
                        %    psdc(:,j,l,(in-1)*ntapers+tp) = psdc(:,j,l,(in-1)*ntapers+tp) + ...
                        %        conj(Jik(:,1,j)).*Jik(:,1,l)  / double(Nwins);
                        %end
                        for l=j:ndim,
                            psdc(:,j,l,(c-1)*ntapers+tp) = psdc(:,j,l,(c-1)*ntapers+tp) + ...
                                conj(Jik(:,1,j)).*Jik(:,1,l) / double(Nwins);
                            if l~=j
                                psdc(:,l,j,(c-1)*ntapers+tp) = conj(psdc(:,j,l,(c-1)*ntapers+tp));
                            end
                        end
                    end
                end
            end
            c = c + 1;
        end
    end
    sumgamma = sumgamma / sum(sumgamma);
    for iNf = 1:Nf
        for indim=1:ndim
            for indim2=1:ndim
                psdc(iNf,indim,indim2,:) = permute(psdc(iNf,indim,indim2,:),[4 1 2 3]) .* sumgamma;
            end
        end
    end
    psd = mean(psdc,4); ipsd = zeros(Nf,ndim,ndim);
    
    for ff=1:Nf, 
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
        for j=1:ndim,
            for l=1:ndim,
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
        [psderr,coherr,pcoherr,pdcerr,sdphase] = spectrerr(psdc,[],coh,pcoh,pdc,options);
    end
    
    if options.rlowess,
        for j=1:ndim,
            psd(:,j,j) = mslowess(f', psd(:,j,j));
            if options.p>0
                psderr(1,:,j,j) = mslowess(f', squeeze(psderr(1,:,j,j))');
                psderr(2,:,j,j) = mslowess(f', squeeze(psderr(2,:,j,j))');
            end
            for l=1:ndim,
                if (options.to_do(1)==1),
                    coh(:,j,l) = mslowess(f', coh(:,j,l));
                    pcoh(:,j,l) = mslowess(f', pcoh(:,j,l));
                end
                if (options.to_do(2)==1),
                    pdc(:,j,l) = mslowess(f', pdc(:,j,l));
                end
                if options.p>0
                    if (options.to_do(1)==1),
                        coherr(1,:,j,l) = mslowess(f', squeeze(coherr(1,:,j,l))');
                        coherr(2,:,j,l) = mslowess(f', squeeze(coherr(2,:,j,l))');
                        pcoherr(1,:,j,l) = mslowess(f', squeeze(pcoherr(1,:,j,l))');
                        pcoherr(2,:,j,l) = mslowess(f', squeeze(pcoherr(2,:,j,l))');
                    end
                    if (options.to_do(2)==1),
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
if sz(1)==1 && sz(2)==2;
    try
        [tapers,eigs]=dpss(N,tapers(1),tapers(2));
    catch 
        error('Problem with dpss - do you have fieldtrip in your path? if so, remove it')
    end
    tapers = tapers*sqrt(Fs);
elseif N~=sz(1);
    error('error in your dpss calculation? the number of time points is different from the length of the tapers');
end;
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
if length(fpass)~=1;
    findx=find(f>=fpass(1) & f<=fpass(end));
else
    [~,findx]=min(abs(f-fpass));
end;
f=f(findx);
end


function J=mtfftc(data,tapers,nfft,Fs)
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

if nargin < 4; error('Need all input arguments'); end;
dtmp=[];
if isstruct(data);
    C=length(data);
    if C==1;
        fnames=fieldnames(data);
        eval(['dtmp=data.' fnames{1} ';'])
        data=dtmp(:);
    end
else
    [N,C]=size(data);
    if N==1 || C==1;
        data=data(:);
    end;
end;
[NC,C]=size(data); % size of data
[NK, K]=size(tapers); % size of tapers
if NK~=NC; error('length of tapers is incompatible with length of data'); end;
tapers=tapers(:,:,ones(1,C)); % add channel indices to tapers
data=data(:,:,ones(1,K)); % add taper indices to data
data=permute(data,[1 3 2]); % reshape data to get dimensions to match those of tapers
data_proj=data.*tapers; % product of data with tapers
J=fft(data_proj,nfft)/Fs;   % fft of projected data
end

function X = loadfile_mt(f,T,options)
if ischar(f)
    if ~isempty(strfind(f,'.mat')), load(f,'X');
    else X = dlmread(f);
    end
else
    X = f;
end
if options.standardise == 1
    for i=1:length(T)
        t = (1:T(i)) + sum(T(1:i-1));
        X(t,:) = X(t,:) - repmat(mean(X(t,:)),length(t),1);
        X(t,:) = X(t,:) ./ repmat(std(X(t,:)),length(t),1);
    end
end
end