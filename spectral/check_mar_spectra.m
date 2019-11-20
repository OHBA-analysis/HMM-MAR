function out = check_mar_spectra( data, T, order, sample_rate, to_plot )
%%function check_mar_spectra( data, T, order, sample_rate )
%
% Function to explore the static fit for a MAR model as a sanity check
% before fitting a HMM-MAR.
%
%
% INPUT
%
% data          observations; a matrix of dimension (nsamples x nchannels)
% T             length of series
% order         model order of MAR
% sample_rate   the sampling frequency of the data
% to_plot       boolean indicating whether to make a summary figure
%
% OUTPUT
%
% mar           structure containing the PSD, r_squared and stabililty of the fitted
% model
%
% Example Usage:
%
% The PSD should be approximately matched to the power estimated by pwelch and pyulear
% with the following options
%
% sample_rate = 250; order = 15;
% mar = hmmmar_check_mar_spectra( data, size(data,1), order, sample_rate );
% [pxx_welch,f_welch] = pwelch( data,hamming(128),[],256,sample_rate, 'psd' );
% [pxx_yule,f_yule] = pyulear( data,order,1024,sample_rate );
%
% figure;subplot(111);grid on; hold on
% plot(f_welch,pxx_welch(:,2),'linewidth',3)
% plot(f_yule,pxx_yule(:,2),'g--','linewidth',2)
% plot(mar.freq_vect,squeeze(abs(mar.PSD(2,2,:))),'r:','linewidth',2)
% legend({'pwelch','pyulear','hmmmar'});
% xlabel('Frequency (Hz)')
% ylabel('Power Spectral Density')

if nargin < 5 || isempty( to_plot )
    to_plot = true;
end

if nargin < 4 || isempty( sample_rate )
    sample_rate = 1;
end

%% Fit AR model

[XX,Y] = formautoregr(data,T,1:order,order,1);
B = XX \ Y;

resids = Y - XX * B;
sigma = cov(Y - XX * B);

nnodes = size(B,2);
delay_vect = 1:order;
A = reshape(B,nnodes,order,nnodes);
A = permute(A,[3,1,2]);

%% Compute Diagnostic Stats and modal decomposition

% Variance explained
r2 = 1 - ( sumsquare(resids) / sumsquare(data) );
r2 = r2*100;

% Create companion form
companion = zeros(nnodes*(order),nnodes*(order));
companion(1:nnodes,:) = B';
companion(nnodes+1:end,1:end-nnodes) = eye(-nnodes + (order)*nnodes);

% modal decomposition
evals = eig(companion);

% resonant frequency per pole
hz = 2*pi ./ angle(evals);
hz = sample_rate ./ hz;

%% Compute the Power Spectral Density

freq_vect = linspace(0,sample_rate/2,1024);
nfreqs = length(freq_vect);

% Fourier transform of parameters
Af = zeros( size(A,1), size(A,2),nfreqs );
iAf = zeros( size(Af) );
% Transfer Function
H = zeros( size(Af) );
% Spectral Matrix
S = zeros( size(Af) );

z = -(1i*2*pi/sample_rate);
for idx = 1:nfreqs
    for ord = 1:order
        Af(:,:,idx) = Af(:,:,idx) + A(:,:,ord).*exp(z .* (delay_vect(ord)) .* freq_vect(idx));
    end
    
    iAf(:,:,idx) = eye(nnodes) - Af(:,:,idx);
    H(:,:,idx) = inv(iAf(:,:,idx));
    
    S(:,:,idx) = H(:,:,idx) * sigma * H(:,:,idx)';
    
end

% Normalise by sample rate
PSD = S ./ (sample_rate ./ 2);

%% Create output object

out = [];
out.PSD = PSD;
out.resid_cov = sigma;
out.freq_vect = freq_vect;
out.stabililty_index = max(abs(evals));
out.r_squared = r2;
out.poles = evals;
out.pole_freq = hz;

%% Make a summary figure
if to_plot == true
    
    figure;
    subplot(231);hold on;grid on;
    plot( sin(linspace(0,2*pi)), cos(linspace(0,2*pi)), 'k' )
    plot( real(evals), imag(evals), 'k.' );
    axis square
    xlabel('Real')
    ylabel('Imaginary')
    title('Poles on z-plane')
    
    subplot(234);hold on;grid on;
    plot( hz, abs(evals), 'k.' );
    xlim( [0, sample_rate/2] )
    axis square
    xlabel('Frequency (Hz)')
    ylabel('Pole Magnitude')
    title('Poles per frequency')
    
    subplot(1,3,[2,3]);hold on;grid on
    for ii = 1:nnodes
        plot( freq_vect, 10*log10(squeeze(abs(PSD(ii,ii,:)))) ,'linewidth',3 );
    end
    ylabel('Power Spectral Density')
    xlabel('Frequency (Hz)')
    title('Power Spectrum per channel')
    xlim([freq_vect(1) freq_vect(end)])
    
    yl = ylim;xl = xlim;
    string = 'Stability Index: %.2f\nR-squared: %.2f%%';
    text( .9*xl(2), .9*yl(2), sprintf(string,max(abs(evals)),r2) );
    
end

function s = sumsquare(a)

s = sum(sum(a.*a));
