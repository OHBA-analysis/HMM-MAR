function [P,Pi] = getMaskedTransProbMats (data,T,hmm,Masks,Gamma,Xi,residuals)
% obtain local Markov transitive probability matrices (LMTPM) for each of the masks
% specified by the variable masks (see description of parameters).
% The matrices include state persistency probabilities 
% (see getTransProbs.m to obtain just the transition probabilities) 
%
% INPUTS:
%
% data          observations - a struct with X (time series) and C (classes)
% T             Number of time points for each time series
% hmm           An hmm structure 
% Masks          A cell where each element is a vector containing the
%               indexes for which we wish to compute the LMTPM; indexes are
%               with respect to the data (not the state time courses, which
%               are typically shorter)
% Gamma         State courses (optional)
% Xi            Joint Prob. of child and parent states given the data (optional)
% residuals     in case we train on residuals, the value of those (optional)
%
% OUTPUTS:
% P             A cell where each element is a LMTPM, computed for the
%               indexes in the corresponding element of Masks
%
% Author: Diego Vidaurre, OHBA, University of Oxford

if nargin<6
    if ~isfield(data,'C')
        if hmm.K>1, data.C = NaN(size(data.X,1),hmm.K);
        else data.C = ones(size(data.X,1),1);
        end
    end
    if nargin<7
        orders = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
        hmm.train.Sind = formindexes(orders,hmm.train.S);
        residuals = getresiduals(data.X,T,hmm.train.Sind,hmm.train.maxorder,hmm.train.order,...
            hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag,hmm.train.zeromean);
    end
    [Gamma,~,Xi]=hsinference(data,T,hmm,residuals);    
end

if ~hmm.train.multipleConf
    [~,order] = formorders(hmm.train.order,hmm.train.orderoffset,hmm.train.timelag,hmm.train.exptimelag);
else
    order = hmm.train.maxorder;
end

N = length(T);
np = length(Masks);
P = cell(1,np); Pi = cell(1,np);
% we do not care about the grouping imposed in the inference
if isfield(hmm.train,'grouping'), hmm.train = rmfield(hmm.train,'grouping'); end

for im = 1:np
    %fprintf('Mask %d \n',im)
    mask = Masks{im};
    T0 = []; Gamma0 = []; Xi0 = [];
    for n = 1:N
        t0 = sum(T(1:n-1)); t1 = sum(T(1:n));
        ind_ix = mask(mask>=t0+1 & mask<=t1); % the ones belonging to this trial
        if length(ind_ix)<=(order+2), continue; end
        T0 = [T0; length(ind_ix)];
        ind_ig = ind_ix(ind_ix>=t0+order+1);
        ind_ig = ind_ig - n*order;
        Gamma0 = cat(1,Gamma0,Gamma(ind_ig,:));
        ind_ixi = ind_ig(1:end-1) - (n-1);
        Xi0 = cat(1,Xi0,Xi(ind_ixi,:,:));
    end
    hmm0 = hsupdate(Xi0,Gamma0,T0,hmm);
    P{im} = hmm0.P; Pi{im} = hmm0.Pi;
end

end
