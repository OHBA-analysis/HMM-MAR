function [options,data,ndim] = checkoptions_spectra (options,data,T,parametric)

if nargin>=3
    if iscell(T)
        T = cell2mat(T); T = T(:);
    end
end

if isfield(options,'pca') && options.pca~=0 && options.pca~=1
    options.pca = 0;
end

if ~isempty(data)
    if iscell(data) && ischar(data{1})
        fsub = data{1};
        loadfile_sub;
        data = X; 
        ndim = size(data,2);
    elseif iscell(data)
        data = data{1};
        ndim = size(data,2);
    elseif isstruct(data)
        ndim = size(data.X,2);
    else
        ndim = size(data,2);
    end
else
    data = rand(10,2); ndim = 2; T = 10; % useless 
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
if ~isfield(options,'verbose'), options.verbose = 0; end
if ~isfield(options,'to_do')
    if ndim>1, options.to_do = ones(2,1); options.to_do(2) = parametric;
    else, options.to_do = zeros(2,1); 
    end
end 
if nargin==3 && any(T<options.win)
    error('The specified window is larger than some of the segments')
end

options.updateGamma = 1; 
options.DirichletDiag = 100; % to avoid a warning later
[options,data] = checkoptions(options,data,[]); 
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
if ~isfield(options,'completelags'), options.completelags = 1; end

end
