function [options,data,ndim] = checkoptions_spectra (options,data,T,parametric)

if nargin==3
    if iscell(T)
        T = cell2mat(T); T = T(:);
    end
end

if isfield(options,'pca') && options.pca~=0 && options.pca~=1
    options.pca = 0;
end

if iscell(data) && ischar(data{1})
    loadfile_sub;
    ndim = size(X,2);
elseif isstruct(data)
    ndim = size(data.X,2);
else
    ndim = size(data,2);
end

% MT and common
if ~isfield(options,'p'), options.p = 0; end
if ~isfield(options,'removezeros'), options.removezeros = 0; end
if ~isfield(options,'completelags'), options.completelags = 0; end
if ~isfield(options,'rlowess'), options.rlowess = 0; end
if ~isfield(options,'numIterations'), options.numIterations = 100; end
if ~isfield(options,'tol'), options.tol = 1e-18; end
if ~isfield(options,'pad'), options.pad = 0; end
if ~isfield(options,'Fs'), options.Fs=1; end
if ~isfield(options,'fpass'),  options.fpass=[options.Fs/200 options.Fs/2]; end
mfs = max(options.Fs/200, options.fpass(1));
if ~isfield(options,'win'),  options.win = min(4*options.Fs/mfs, min(T));  end
if ~isfield(options,'tapers'), options.tapers = [4 7]; end
if ~isfield(options,'to_do')
    if ndim>1, options.to_do = ones(2,1); options.to_do(2) = parametric;
    else, options.to_do = zeros(2,1); 
    end
end 
if nargin==3 && any(T<options.win)
    error('The specified window is larger than some of the segments')
end

[options,data] = checkoptions(options,data,T);
options.leida = 0;
options.onpower = 0;
options.pca = 0;
options.pca_spatial = 0;
options.embeddedlags = 0;
if isfield(options,'As'), options = rmfield(options,'As'); end
if isfield(options,'A'), options = rmfield(options,'A'); end
if isstruct(data), data = data.X; end

% MAR 
% if ~isfield(options,'loadings'), options.loadings=eye(ndim); end;
if ~isfield(options,'Nf'),  options.Nf=256; end
if ~isfield(options,'MLestimation'), options.MLestimation = 1; end
if ~isfield(options,'completelags'), options.completelags = 0; end

if ~isfield(options,'level'), options.level = 'group'; end
if strcmp(options.level,'subject') && options.p>0
   warning('Intervals of confidence can only be computed when subject.level is group; setting options.p=0 ...')
   options.p = 0;
end

end
