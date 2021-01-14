function ord = plot_vpath (vpath,T,continuous,order_trials,behaviour,cm)
  
if nargin < 2 || isempty(T), T = size(vpath,1); end
if nargin < 3, continuous = length(T)==1 || any(T(1)~=T); end % show it as continuous data? 
if nargin < 4, order_trials = false; end % order the trials?
if nargin < 5, behaviour = []; end % behaviour with respect to which order the trials
if nargin < 6, cm = colormap; end % colormap

Gamma = vpath_to_stc(vpath);
ord = plot_Gamma (Gamma,T,continuous,order_trials,behaviour,cm);

end
