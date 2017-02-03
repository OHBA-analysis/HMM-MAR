function avfit = get_spectrabybands(fit,bands)
% Divides the spectral information (psd,coh,pcoh,pdc) in parameter fit by bands;
% Parameter bands has one row per band, and two columns which indicate the
% beginning and the end of each band

K = length(fit.state);
N = size(bands,1); 
ndim = size(fit.state(1).psd,2);
f = fit.state(1).f;
avfit = fit;

for k=1:K
    
    avfit.state(k).psd = zeros(N,ndim,ndim);
    for j=1:N
        ind = (f>=bands(j,1)) & (f<=bands(j,2));
        avfit.state(k).psd(j,:,:) = sum(fit.state(k).psd(ind,:,:));
    end
    
    if isfield(fit.state(1),'coh')
        avfit.state(k).coh = zeros(N,ndim,ndim);
        for j=1:N
            ind = (f>=bands(j,1)) & (f<=bands(j,2));
            avfit.state(k).coh(j,:,:) = sum(fit.state(k).coh(ind,:,:));
        end
    end
    
    if isfield(fit.state(1),'pcoh')
        avfit.state(k).pcoh = zeros(N,ndim,ndim);
        for j=1:N
            ind = (f>=bands(j,1)) & (f<=bands(j,2));
            avfit.state(k).pcoh(j,:,:) = sum(fit.state(k).pcoh(ind,:,:));
        end
    end    
    
    if isfield(fit.state(1),'pdc')
        avfit.state(k).pdc = zeros(N,ndim,ndim);
        for j=1:N
            ind = (f>=bands(j,1)) & (f<=bands(j,2));
            avfit.state(k).pdc(j,:,:) = sum(fit.state(k).pdc(ind,:,:));
        end
    end 
    
end

avfit.state = rmfield(avfit.state,'f');
if isfield(avfit.state(1),'ipsd')
    avfit.state = rmfield(avfit.state,'ipsd');
end
if isfield(avfit.state(1),'phase')
    avfit.state = rmfield(avfit.state,'phase');
end
if isfield(avfit.state(1),'psderr')
    avfit.state = rmfield(avfit.state,'psderr');
end
if isfield(avfit.state(1),'coherr')
    avfit.state = rmfield(avfit.state,'coherr');
end
if isfield(avfit.state(1),'pdcerr')
    avfit.state = rmfield(avfit.state,'pdcerr');
end
if isfield(avfit.state(1),'pcoherr')
    avfit.state = rmfield(avfit.state,'pcoherr');
end
if isfield(avfit.state(1),'sdphase')
    avfit.state = rmfield(avfit.state,'sdphase');
end

avfit.bands = bands;