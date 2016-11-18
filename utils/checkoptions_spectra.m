function [options,Gamma] = checkoptions_spectra (options,ndim,T)

if nargin==3
    if iscell(T)
        T = cell2mat(T); T = T(:);
    end
end

% MT and common
if ~isfield(options,'p'), options.p = 0; end
if ~isfield(options,'removezeros'), options.removezeros = 0; end
if ~isfield(options,'completelags'), options.completelags = 0; end
if ~isfield(options,'rlowess'), options.rlowess = 0; end
if ~isfield(options,'numIterations'), options.numIterations = 100; end
if ~isfield(options,'tol'), options.tol = 1e-18; end
if ~isfield(options,'pad'), options.pad = 0; end;
if ~isfield(options,'Fs'), options.Fs=1; end;
if ~isfield(options,'fpass'),  options.fpass=[0 options.Fs/2]; end;
if ~isfield(options,'tapers'), options.tapers = [4 7]; end;
if ~isfield(options,'win'), options.win = min(T);  end
if ~isfield(options,'standardise'), options.standardise = 0; end
if ~isfield(options,'to_do'), 
    if ndim>1, options.to_do = ones(2,1); 
    else options.to_do = zeros(2,1); 
    end
end

if nargin==3 && any(T<options.win)
    error('The specified window is larger than some of the segments')
end

% MAR 
% if ~isfield(options,'loadings'), options.loadings=eye(ndim); end;
if ~isfield(options,'Nf'),  options.Nf=256; end;
if ~isfield(options,'MLestimation'), options.MLestimation = 1; end
if ~isfield(options,'completelags'), options.completelags = 0; end

if ~isfield(options,'level'), options.level = 'group'; end

% if options.MLestimation == 0, 
%     error('options.MLestimation equal to 0 (Bayesian) is not currently an option')
% end

if strcmp(options.level,'subject') && options.p>0
   warning('Intervals of confidence can only be computed when subject.level is group; setting options.p=0 ...')
   options.p = 0;
end

if nargout==2
    if ~isfield(options,'Gamma'),
        Gamma = ones(sum(T),1);
    else
        Gamma = options.Gamma;
    end
end

end
