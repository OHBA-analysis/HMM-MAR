function [psd_tf,coh_tf,pdc_tf] = hmmtimefreq(spectra,Gamma,center)
% obtains a time-frequency representation of the power spectra density,
% coherence and PDC - the estimations of which (contained in fit) had to be
% estimated with hmmspectramar or hmmspectramt. If coherence or PDC are
% missing from the estimation, the corresponding output arguments are
% empty. The argument Gamma contains the state time courses. 
% If the third argument is 1, then each frequency bin is centered 
%
% The output arguments are
%  psd_tf: (time points by no. of frequency bins by no.regions) 
%  coh_tf: (time points by no. of frequency bins by no.regions by no.regions) 
%  pdc_tf: (time points by no. of frequency bins by no.regions by no.regions)
%
% To display later: imagesc(time,freq,psd_tf(:,:,1)')  
%  i.e. you need to transpose
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2016)

if nargin<3, center = 0; end

coh_tf = []; pdc_tf = [];

[T,K] = size(Gamma);
nf = length(spectra.state(1).f);
ndim = size(spectra.state(1).psd,2);

psd_tf = zeros(T,nf,ndim);
for k=1:K
    for n=1:ndim
        psd_tf(:,:,n) = psd_tf(:,:,n) + repmat(Gamma(:,k),[1 nf]) .* ...
            repmat(spectra.state(k).psd(:,n,n),[1 T])';
    end
end
if center
    for f=1:nf
        for n=1:ndim
            psd_tf(:,f,n) = psd_tf(:,f,n) - mean(psd_tf(:,f,n));
        end
    end
end

if isfield(spectra.state(1),'coh')
    coh_tf = ones(T,nf,ndim,ndim);
    for k=1:K
        for n1=1:ndim-1
            for n2=n1+1:ndim
                coh_tf(:,:,n1,n2) = coh_tf(:,:,n1,n2) + repmat(Gamma(:,k),[1 nf]) .* ...
                    repmat(spectra.state(k).coh(:,n1,n2),[1 T])';
                coh_tf(:,:,n2,n1) = coh_tf(:,:,n1,n2);
            end
        end
    end
    if center
        for f=1:nf
            for n1=1:ndim-1
                for n2=n1+1:ndim
                    coh_tf(:,f,n1,n2) = coh_tf(:,f,n1,n2) - mean(coh_tf(:,f,n1,n2));
                    coh_tf(:,f,n2,n1) = coh_tf(:,f,n1,n2);
                end
            end
        end
    end
end

if isfield(spectra.state(1),'pdc')
    pdc_tf = ones(T,nf,ndim,ndim);
    for k=1:K
        for n1=1:ndim
            for n2=1:ndim
                if n1==n2, continue; end
                pdc_tf(:,:,n1,n2) = pdc_tf(:,:,n1,n2) + repmat(Gamma(:,k),[1 nf]) .* ...
                    repmat(spectra.state(k).pdc(:,n1,n2),[1 T])';
            end
        end
    end
    if center
        for f=1:nf
            for n1=1:ndim
                for n2=1:ndim
                    if n1==n2, continue; end
                    pdc_tf(:,f,n1,n2) = pdc_tf(:,f,n1,n2) - mean(pdc_tf(:,f,n1,n2));
                end
            end
        end
    end
end

end