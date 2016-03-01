function [fit,psdc,corrc] = gammaspectra(Gamma,T,options)
% Get ML spectral estimates of the probabilities 
%   sum_k Prob(Gamma(t)==k AND Gamma(t-lag)==k)
%  
%
% INPUT
% Gamma         State time course (or Viterbi path) 
% T             Number of time points for each time series
% options 

%  .Fs:       Sampling frequency
%  .fpass:    Frequency band to be used [fmin fmax] (default [0 fs/2])
%  .p:        p-value for computing jackknife confidence intervals (default 0)
%  .Nf        No. of frequencies to be computed in the range minHz-maxHz
%  .nlags     Maximum no. of lags to consider
%
% OUTPUT
% fit.psd     [Nf x 1] Power Spectral Density of these probabilities
% fit.psderr     [2 x Nf] Interval of confidence
% fit.state(k).coh     [Nf x N x N] Coherence matrix
% fit.state(k).f     [Nf x 1] Frequency vector
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2016)

% it is a Viterbi Path
if size(Gamma,2) == 1
    VP = Gamma;
    states = unique(VP);
    K = length(states);
    Gamma = zeros(size(VP,1),K,'single');
    for k=1:K
        Gamma(VP==states(k),k) = 1;
    end
    clear VP
end

order = (sum(T) - size(Gamma,1)) / length(T);
T = T - order; 
N = length(T);

if ~isfield(options,'Fs'), 
    error('You need to specify the sampling rate in options.Fs'); 
end
if isfield(options,'nlags'), nlags = options.nlags; 
else nlags = 50; 
end;
if isfield(options,'Nf'), Nf = options.Nf; 
else Nf = 256; 
end;
seconds = nlags / options.Fs;
if isfield(options,'fpass'), fpass = options.fpass;
else fpass = [1/seconds options.Fs/2]; 
end;
if ~isfield(options,'p'), options.p = 0; end
 
if N<5 && options.p>0,  
    error('You need at least 5 trials to compute error bars for MAR spectra'); 
end

freqs = (0:Nf-1)*( (fpass(2) - fpass(1)) / (Nf-1)) + fpass(1);
w = 2*pi*freqs/options.Fs;

corrc = zeros(nlags,N);
% get mean lagged probabilities
for i=1:N
    t0 = sum(T(1:i-1)); 
    Gammaj = Gamma(t0+1:t0+T(i),:); Tj = T(i);
    for j = 1:nlags
        corrc(j,i) = mean(sum(Gammaj(1:Tj-nlags,:) .* Gammaj(1+j:Tj-nlags+j,:),2));
    end
end
corrc = [corrc(end:-1:1,:); ones(1,N); corrc];
   
psdc = zeros(Nf,N);
% fourier on mean lagged probabilities
for ff=1:Nf,
    for i=1:N % Wiener?Khinchin theorem
        psdc(ff,i) = exp(-1i*w(ff)*(-nlags:nlags)) * corrc(:,i);
    end
end
psdc = abs(psdc);

fit = struct();
fit.psd = mean(psdc,2);
fit.corr = mean(corrc,2);
if options.p>0 % jackknife estimation
    psd_jackknife = zeros(Nf,N);
    corr_jackknife = zeros(2*nlags+1,N);
    for i = 1:N
        ind = setdiff(1:N,i);
        psd_jackknife(:,i) = mean(psdc(:,ind),2);
        corr_jackknife(:,i) = mean(corrc(:,ind),2);
    end
    dof = N-1;
    tcrit = tinv(1-options.p/2,dof); % Inverse of Student's T cumulative distribution function
    sigma = tcrit * sqrt(dof)*std(log(psd_jackknife),1,2);
    fit.psderr(1,:) = fit.psd .* exp(-sigma);
    fit.psderr(2,:) = fit.psd .* exp(sigma);
    sigma = tcrit * sqrt(dof)*std(log(corr_jackknife),1,2);
    fit.correrr(1,:) = fit.corr .* exp(-sigma);
    fit.correrr(2,:) = fit.corr .* exp(sigma);    
end
fit.f = freqs;
fit.t = (1:nlags)/options.Fs;


end

