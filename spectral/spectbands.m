function sp_fit = spectbands(sp_fit,bands)
%
% From a multitaper or MAR estimation of the states spectra,  
% it decomposes the spectral information (which is given by frequency bin)
% into a set of frequency bands in order to facilitate interpretation 
% and visualisation

%
% INPUTS: 
%
% sp_fit                The output of hmmspectramar or hmmspectramt. 
% bands                 Matrix with one row per band, and two columns 
%                       indicating the beginning and the end of each band
% 
% OUTPUT:
% 
% sp_fit                A struct with the spectral information across bands,
%                        with the same fields and dimensions that the input 'sp_fit', 
%                       but with (no. of bands) components instead of No. frequency bins.
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2017)

K = length(sp_fit.state);
Nbands = size(bands,1); 
ndim = size(sp_fit.state(1).psd,2);
f = sp_fit.state(1).f;
av_fit = sp_fit;

for k=1:K
    
    av_fit.state(k).psd = zeros(Nbands,ndim,ndim);
    for j=1:Nbands
        ind = (f>=bands(j,1)) & (f<=bands(j,2));
        av_fit.state(k).psd(j,:,:) = sum(sp_fit.state(k).psd(ind,:,:));
    end
    
    if isfield(sp_fit.state(1),'coh')
        av_fit.state(k).coh = zeros(Nbands,ndim,ndim);
        for j=1:Nbands
            ind = (f>=bands(j,1)) & (f<=bands(j,2));
            av_fit.state(k).coh(j,:,:) = sum(sp_fit.state(k).coh(ind,:,:));
        end
    end
    
    if isfield(sp_fit.state(1),'pcoh')
        av_fit.state(k).pcoh = zeros(Nbands,ndim,ndim);
        for j=1:Nbands
            ind = (f>=bands(j,1)) & (f<=bands(j,2));
            av_fit.state(k).pcoh(j,:,:) = sum(sp_fit.state(k).pcoh(ind,:,:));
        end
    end    
    
    if isfield(sp_fit.state(1),'pdc')
        av_fit.state(k).pdc = zeros(Nbands,ndim,ndim);
        for j=1:Nbands
            ind = (f>=bands(j,1)) & (f<=bands(j,2));
            av_fit.state(k).pdc(j,:,:) = sum(sp_fit.state(k).pdc(ind,:,:));
        end
    end 
    
end

av_fit.state = rmfield(av_fit.state,'f');
if isfield(av_fit.state(1),'ipsd')
    av_fit.state = rmfield(av_fit.state,'ipsd');
end
if isfield(av_fit.state(1),'phase')
    av_fit.state = rmfield(av_fit.state,'phase');
end
if isfield(av_fit.state(1),'psderr')
    av_fit.state = rmfield(av_fit.state,'psderr');
end
if isfield(av_fit.state(1),'coherr')
    av_fit.state = rmfield(av_fit.state,'coherr');
end
if isfield(av_fit.state(1),'pdcerr')
    av_fit.state = rmfield(av_fit.state,'pdcerr');
end
if isfield(av_fit.state(1),'pcoherr')
    av_fit.state = rmfield(av_fit.state,'pcoherr');
end
if isfield(av_fit.state(1),'sdphase')
    av_fit.state = rmfield(av_fit.state,'sdphase');
end

sp_fit = av_fit; 
sp_fit.bands = bands;

end