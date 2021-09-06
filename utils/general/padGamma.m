function Gamma = padGamma(Gamma,T,options)
% Unless a Gaussian distribution is used to describe the states, the state
% time courses are shorter than the  data time series.
% This function adjusts the state time courses to have the same size than
% the data time series, padding the state time courses with probabilities
% that correspond to the trial-specific fractional occupancies,
% and upsampling in case options.downsample was specified
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2017)

% do_upsample = (isfield(options,'downsample') && options.downsample > 0); 
do_upsample = false; % this doesn't work always so deactivate

do_chop = 0;
if isfield(options,'embeddedlags') && length(options.embeddedlags) > 1
    d = [ -min(options.embeddedlags) max(options.embeddedlags) ]; do_chop = 1;
elseif isfield(options,'order') && options.order > 1 
    d = [options.order 0]; do_chop = 1;
end

if ~do_chop && ~do_upsample, return; end % nothing to do

if iscell(T)
    if size(T,1)==1, T = T'; end
    T = cell2mat(T);
end

K = size(Gamma,2); offset = sum(d); N = length(T); Tshifted = T - offset;

if do_upsample
    r = options.downsample / options.Fs;
    Gamma_orig = Gamma;
    Gamma = zeros(sum(Tshifted),K);
    Tdown = ceil(r * Tshifted );
    for j = 1:N
        t1 = (1:Tdown(j)) + sum(Tdown(1:j-1));
        t2 = (1:Tshifted(j)) + sum(Tshifted(1:j-1));
        for k = 1:K
            g = resample(Gamma_orig(t1,k), options.Fs, options.downsample);
            if length(g)>length(t2), Gamma(t2,k) = g(1:end-1);
            else,  Gamma(t2,k) = g;
            end
        end
    end
end

if do_chop
    Gamma_orig = Gamma;
    Gamma = zeros(sum(T),K);
    for j = 1:N
        t1 = (1:Tshifted(j)) + sum(Tshifted(1:j-1));
        t2 = (1:T(j)) + sum(T(1:j-1));
        mg = mean(Gamma_orig(t1,:));
        Gamma(t2,:) = [ repmat(mg,d(1),1); Gamma_orig(t1,:); repmat(mg,d(2),1) ];
    end
end



end