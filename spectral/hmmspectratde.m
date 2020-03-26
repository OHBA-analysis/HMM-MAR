function fit = hmmspectratde(hmm,options)
% Get spectral estimates from a time-delay embedded model
%
% INPUT
%
% hmm           An hmm-mar structure 
%
% OUTPUT
% fit is a list with K elements, each of which contains: 
% fit.state(k).psd     (Nf x ndim x ndim) Power Spectral Density matrix
% fit.state(k).ipsd     (Nf x ndim x ndim) Inverse Power Spectral Density matrix
% fit.state(k).coh     (Nf x ndim x ndim) Coherence matrix
% fit.state(k).pcoh    (Nf x ndim x ndim) Partial Coherence matrix
% fit.state(k).phase     (Nf x ndim x ndim) Phase matrix
% fit.state(k).f     (Nf x 1) Frequency vector
%       (where ndim is the number of channels) 
%

warning('The use of this function for computing the spectra is not recommended, specially if options.pca was used') 

L = length(hmm.train.embeddedlags);
if L==1, error('Not a TDE model'); end 

options.K = length(hmm.state);
options = checkoptions_spectra(options,[],[],1);

C = getAutoCovMat(hmm,1,0);
ndim = size(C,1) / L;
K = length(hmm.state); 

if isfield(options,'Fs')
    Fs = options.Fs; 
elseif isfield(hmm.train,'downsample') && hmm.train.downsample~=0
    Fs = hmm.train.downsample;
else
    Fs = hmm.train.Fs;
end

% freqs = (0:options.Nf-1)* ...
%     ( (options.fpass(2) - options.fpass(1)) / (options.Nf-1)) + options.fpass(1);
% w = 2*pi*freqs/Fs;

Nf = round(options.Nf * (Fs/2) / (options.fpass(2) - options.fpass(1)));

freqs = linspace(-Fs/2,Fs/2,2*Nf+1);
freqs = freqs(Nf+2:end);
indf = (freqs>=options.fpass(1) & freqs<=options.fpass(2));
freqs = freqs(indf);

%ind = 1:L:ndim*L;

for k = 1:K
    fit.state(k).f = freqs; 
    fit.state(k).psd = zeros(Nf,ndim,ndim);
    fit.state(k).ipsd = zeros(Nf,ndim,ndim);
    fit.state(k).coh = zeros(Nf,ndim,ndim);
    fit.state(k).pcoh = zeros(Nf,ndim,ndim);
    fit.state(k).phase = zeros(Nf,ndim,ndim);
    
    C = getAutoCovMat(hmm,k,0);
    %C = C(end:-1:1,:);
    
    for j1 = 1:ndim
        for j2 = 1:ndim
            indj1 = (1:L) + (j1-1)*L;
            indj2 = (1:L) + (j2-1)*L;
            Cj = C(indj1,indj2);
            c = diag(Cj(end:-1:1,:));
            psd = fftshift(fft(c,2*Nf+1));
            fit.state(k).psd(:,j1,j2) = psd(Nf+2:end);
        end
    end
        
%     for ff = 1:options.Nf
%         A = zeros(ndim);
%         for i = 1:L
%             o = hmm.train.embeddedlags(i);
%             ind_i = ind + (i-1);
%             A = A + C(ind_i,ind_i) * exp(-1i*w(ff)*o);
%         end
%         fit.state(k).psd(ff,:,:) = A;
%     end
    for ff = 1:options.Nf
        fit.state(k).ipsd(ff,:,:) = inv(permute(fit.state(k).psd(ff,:,:),[3 2 1]));
    end
    for n1 = 1:ndim
       for n2 = 1:ndim
           rkj = fit.state(k).psd(:,n1,n2)./(sqrt(fit.state(k).psd(:,n1,n1)).*sqrt(fit.state(k).psd(:,n2,n2)));
           fit.state(k).coh(:,n1,n2) = abs(rkj);
           fit.state(k).pcoh(:,n1,n2) = -fit.state(k).ipsd(:,n1,n2)./...
               (sqrt(fit.state(k).ipsd(:,n1,n1)).*sqrt(fit.state(k).ipsd(:,n2,n2)));
           fit.state(k).phase(:,n1,n2) = atan(imag(rkj)./real(rkj));
       end
    end
    for n = 1:ndim
        fit.state(k).psd(:,n,n) = abs(fit.state(k).psd(:,n,n));
    end
    
    fit.state(k).psd = fit.state(k).psd(indf,:,:);
    fit.state(k).ipsd = fit.state(k).ipsd(indf,:,:);
    fit.state(k).coh = fit.state(k).coh(indf,:,:);
    fit.state(k).pcoh = fit.state(k).pcoh(indf,:,:);
    fit.state(k).phase = fit.state(k).phase(indf,:,:);
    
end

end

