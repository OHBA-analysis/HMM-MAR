function vp = padVP(vp,T,options)
% Unless a Gaussian distribution is used to describe the states, the state
% time courses are shorter than the  data time series.
% This function adjusts the Viterbi path to have the same size than
% the data time series, padding the Viterbi Path with NaN. 
% 
% Author: Diego Vidaurre, OHBA, University of Oxford (2019)


if isfield(options,'embeddedlags') && length(options.embeddedlags) > 1
    d = [ -min(options.embeddedlags) max(options.embeddedlags) ];
elseif isfield(options,'order') && options.order > 1 
    d = [options.order 0];
else
    return
end

if iscell(T)
    if size(T,1)==1, T = T'; end
    T = cell2mat(T);
end

if isfield(options,'downsample') && options.downsample > 0
    r =  options.downsample / options.Fs;
    T = ceil(r * T); 
end

offset = sum(d); N = length(T);

vp_orig = vp; 
vp = zeros(sum(T),1);

for j = 1:N
    t1 = (1:T(j)-offset) + sum(T(1:j-1)) - (j-1)*offset; 
    t2 = (1:T(j)) + sum(T(1:j-1));
    vp(t2) = [ repmat(NaN,d(1),1); vp_orig(t1); repmat(NaN,d(2),1) ];
end

end