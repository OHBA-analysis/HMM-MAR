function sp_fit = relspect(sp_fit,standardise)
%
% Computes the relative spectra for a number of states' estimations.
% This is done by substracting, for each frequency bin (or frequency
% component, as returned by spectdecompose) and region, the mean of the
% estimation across states.
%
%
% INPUTS:
%
% sp_fit                The output of hmmspectramar or hmmspectramt
% standardise           Whether or not to divide by the standard deviation
%                       across states as well as demeaning (default 0)
%
% OUTPUT:
%
% sp_fit                A struct with the spectral information
%                       with the same fields and dimensions that the input 'sp_fit'
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2017)

if nargin < 2, standardise = 0; end

K = length(sp_fit.state);
Nf = size(sp_fit.state(1),1);
ndim = size(sp_fit.state(1).psd,2);
av_fit = sp_fit;

m = zeros(Nf,ndim,ndim);
s = zeros(Nf,ndim,ndim);

% PSD
for k = 1:K
    m = abs(av_fit.state(k).psd) / K + m;
    s = abs(av_fit.state(k).psd).^2 / K + m;
end
for k = 1:K
    av_fit.state(k).psd = abs(av_fit.state(k).psd) - m;
    if standardise
        av_fit.state(k).psd = abs(av_fit.state(k).psd) ./ s;
    end
end

% Coh
if isfield(av_fit.state(k),'coh')
    for k = 1:K
        m = abs(av_fit.state(k).coh) / K + m;
        s = abs(av_fit.state(k).coh).^2 / K + m;
    end
    for k = 1:K
        av_fit.state(k).coh = abs(av_fit.state(k).coh) - m;
        if standardise
            av_fit.state(k).coh = abs(av_fit.state(k).coh) ./ s;
        end
        for f = 1:Nf
            av_fit.state(k).coh(f,:,:) = eye(ndim);
        end
    end
end

% PDC
if isfield(av_fit.state(k),'pdc')
    for k = 1:K
        m = abs(av_fit.state(k).pdc) / K + m;
        s = abs(av_fit.state(k).pdc).^2 / K + m;
    end
    for k = 1:K
        av_fit.state(k).coh = abs(av_fit.state(k).coh) - m;
        if standardise
            av_fit.state(k).coh = abs(av_fit.state(k).coh) ./ s;
        end
        for f = 1:Nf
            av_fit.state(k).pdc(f,:,:) = eye(ndim);
        end
    end
end

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

end