function [Path,Xi] = hmmsdecode(Xin,T,hmm,type)
% Computes the state time courses or viterbi paths
%
% INPUTS
% Xin: cell with strings referring to the subject files
% T: cell of vectors, where each element has the length of each trial per
% hmm: the stochastic HMM structure
% type: 0, state time courses; 1, viterbi path
% NOTE: computations of stats now done in getshmmstats.m
%
% Diego Vidaurre, OHBA, University of Oxford (2015)

if nargin<4, type = 0; end

for i = 1:length(T)
    if size(T{i},1)==1, T{i} = T{i}'; end
end

N = length(Xin);
K = length(hmm.state);
TT = []; for i=1:N, TT = [TT; T{i}]; end

if length(hmm.train.embeddedlags)>1
    L = -min(hmm.train.embeddedlags) + max(hmm.train.embeddedlags);
    maxorder = 0;
else
    L = hmm.train.maxorder;
    maxorder = hmm.train.maxorder; 
end

if hmm.train.downsample > 0
    downs_ratio = (hmm.train.downsample/hmm.train.Fs);
else
    downs_ratio = 1;
end

Xi = [];
if type==0
    Path = zeros(downs_ratio * (sum(TT)-length(TT)*L),K,'single');
    if nargout>=2
        Xi = zeros(downs_ratio * (sum(TT)-length(TT)*(L+1)),K,K,'single');
    end
else
    Path = zeros(downs_ratio * (sum(TT)-length(TT)*L),1,'single');
    if nargout>=2, Xi = []; end
end

tacc = 0; tacc2 = 0;
for i = 1:N
    [X,XX,Y,Ti] = loadfile(Xin{i},T{i},hmm.train);
    hmm_i = hmm;
    hmm_i.train.embeddedlags = 0;
    hmm_i.train.pca = 0;
    hmm_i.train.pca_spatial = 0;
    if isfield(hmm_i.train,'BIGNbatch')
        hmm_i.train = rmfield(hmm_i.train,'BIGNbatch');
    end
    t = (1:(sum(Ti)-length(Ti)*maxorder)) + tacc;
    t2 = (1:(sum(Ti)-length(Ti)*(maxorder+1))) + tacc2;
    tacc = tacc + length(t); tacc2 = tacc2 + length(t2);
    if type==0
        data = struct('X',X,'C',NaN(sum(Ti)-length(Ti)*maxorder,K));
        [gamma,~,xi] = hsinference(data,Ti,hmm_i,Y,[],XX);
        checkGamma(gamma,Ti,hmm_i.train,i);
        Path(t,:) = single(gamma);
        if nargout>=2, Xi(t2,:,:) = xi; end
    else
        Path(t,:) = hmmdecode(X,Ti,hmm_i,type,Y,0);
    end
    
end


end
