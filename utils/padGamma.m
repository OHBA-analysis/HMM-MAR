function Gamma = padGamma(Gamma,T,options)
% Unless a Gaussian distribution is used to describe the states, the state
% time courses are shorter than the  data time series.
% This function adjusts the state time courses to have the same size than
% the data time series, padding the state time courses with probabilities
% that correspond to the trial-specific fractional occupancies. 
% 
% Author: Diego Vidaurre, OHBA, University of Oxford (2017)


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

K = size(Gamma,2); offset = sum(d); N = length(T);

Gamma_orig = Gamma; 
Gamma = zeros(sum(T),K);

for j = 1:N
    t1 = (1:T(j)-offset) + sum(T(1:j-1)) - (j-1)*offset; 
    t2 = (1:T(j)) + sum(T(1:j-1));
    mg = mean(Gamma_orig(t1,:));
    Gamma(t2,:) = [ repmat(mg,d(1),1); Gamma_orig(t1,:); repmat(mg,d(2),1) ];
end

end