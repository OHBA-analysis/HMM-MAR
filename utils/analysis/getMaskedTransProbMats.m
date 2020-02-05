function [P,Pi] = getMaskedTransProbMats (data,T,hmm,Masks,Gamma,Xi)
% Obtain local Markov transitive probability matrices (LMTPM) for each of the 
% masks specified by the variable masks (see description of parameters).
% The matrices include state persistency probabilities 
% (see getTransProbs.m to obtain just the transition probabilities) 
%
% INPUTS:
%
% data          observations; either a struct with X (time series) and C (classes, optional),
%                             or a matrix containing the time series,
%                             or a list of file names
% T             Number of time points for each time series
% hmm           An hmm structure 
% Masks         A cell where each element is a vector containing the indexes
%               for which we wish to compute the LMTPM.
%               For example, if Masks is {[1001:2000],[2001:5000]}, then  
%               P{1} and Pi{1} will be computed for time points between
%               1001 to 2000; and P{2} and Pi{2} will be computed for time
%               points 2001 to 5000. This way, it is possible,for instance, 
%               to compute a separate LMTPM for each session or trial. 
%               Note that the indexes are with respect to the data, not to
%               the state time courses which can be shorter if
%               options.order or options.embeddedlags were used. For
%               example, if options.order=2 was used, the state time courses (Gamma) 
%               for each segment will have 2 fewer time points.
% Gamma         State courses (optional) - will be recomputed if not provided
% Xi            Joint Prob. of child and parent states given the data (optional)
%
% OUTPUTS:
% P             A cell where each element is a LMTPM, computed for the
%               indexes in the corresponding element of Masks
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2017)

if nargin<6 || isempty(Xi)
    options = hmm.train;
    options.updateGamma = 1; 
    options.updateP = 0; 
    options.updateObs = 0; 
    options.verbose = 0;
    if isfield(options,'Gamma'), options = rmfield(options,'Gamma'); end
    if isfield(options,'orders'), options = rmfield(options,'orders'); end
    if isfield(options,'active'), options = rmfield(options,'active'); end
    options.hmm = hmm;
    [hmm, Gamma, Xi] = hmmmar (data,T,options);  
end

order = hmm.train.maxorder;
embeddedlags = abs(hmm.train.embeddedlags); 
L = order + embeddedlags(1) + embeddedlags(end);

if ~iscell(Masks), Masks = {Masks}; end
N = length(T);
np = length(Masks);
P = cell(1,np); Pi = cell(1,np);
% we do not care about the grouping imposed in the inference
if isfield(hmm.train,'grouping'), hmm.train = rmfield(hmm.train,'grouping'); end
if ~isfield(hmm.train,'Pstructure'), hmm.train.Pstructure = true(hmm.K); end
if ~isfield(hmm.train,'Pistructure'), hmm.train.Pistructure = true(1,hmm.K); end

for im = 1:np
    %fprintf('Mask %d \n',im)
    mask = Masks{im};
    T0 = []; Gamma0 = []; Xi0 = [];
    for n = 1:N
        t0 = sum(T(1:n-1)); t1 = sum(T(1:n));
        ind_ix = mask(mask>=t0+1 & mask<=t1); % the ones belonging to this trial
        if ~isempty(ind_ix) && ind_ix(end)<t1, t1=ind_ix(end); end % control for masks cutting off trials
        if length(ind_ix)<=L, continue; end
        T0 = [T0; length(ind_ix)];
        if order > 0
            ind_ig = ind_ix(ind_ix>=t0+order+1);
            ind_ig = ind_ig - n*order;
        elseif length(embeddedlags) > 1
            ind_ig = ind_ix((ind_ix>=t0+embeddedlags(1)+1) & (ind_ix<=t1-embeddedlags(end))  ); 
            ind_ig = ind_ig - (n-1)*L - embeddedlags(1);
        else 
            ind_ig = ind_ix;
        end
        ind_ixi = ind_ig(1:end-1) - (n-1);    
        Gamma0 = cat(1,Gamma0,Gamma(ind_ig,:));
        Xi0 = cat(1,Xi0,Xi(ind_ixi,:,:));
    end
    if isempty(Gamma0), error('Invalid mask?'); end
    hmm0 = hsupdate(Xi0,Gamma0,T0,hmm);
    P{im} = hmm0.P; Pi{im} = hmm0.Pi;
end

end
