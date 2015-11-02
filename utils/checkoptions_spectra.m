function [options,Gamma] = checkoptions_spectra (options,ndim,T)

if ~isfield(options,'Gamma'),
    %if ~isfield(options,'order'), error('If you do not supply Gamma, you need to provide order'); end
    Gamma = ones(sum(T),1);
else
    Gamma = options.Gamma;
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
if ~isfield(options,'win'), options.win = min(T); end
if ~isfield(options,'to_do'), options.to_do = ones(2,1); end;

if any(T<options.win),
    error('The specified window is larger than some of the segments')
end

% MAR 
if ~isfield(options,'loadings'), options.loadings=eye(ndim); end;
if ~isfield(options,'Nf'),  options.Nf=256; end;

end
