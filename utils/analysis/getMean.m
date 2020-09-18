function m = getMean(hmm,k,original_space)
% Get the mean activity for state k from the estimated model hmm
% This function should primarily be used when the state model is a Gaussian
% distribution (i.e. order=0)
%
% Diego Vidaurre, OHBA, University of Oxford (2016)


if hmm.train.zeromean
    error('The states are not modelled by the mean (zeromean=1)')
end
if hmm.train.order > 0
    warning(['This is a MAR model, you might to run some spectral analysis ' ...
        'and look at the PSD (power) instead'])
end
if nargin < 3, original_space = true; end

if nargin < 2 || isempty(k)
    m = [];
    for k = 1:length(hmm.state)
        mk = hmm.state(k).W.Mu_W(1,:);
        if isfield(hmm.train,'A') && original_space
            mk = mk * hmm.train.A';
        end
        m = [m; mk];
    end
    m = m'; 
else
    m = hmm.state(k).W.Mu_W(1,:);
    if isfield(hmm.train,'A') && original_space
        m = m * hmm.train.A';
    end
end
end