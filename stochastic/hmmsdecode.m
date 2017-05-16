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
tacc = 0; tacc2 = 0;

if length(hmm.train.embeddedlags)>1
    L = -min(hmm.train.embeddedlags) + max(hmm.train.embeddedlags);
else
    L = hmm.train.maxorder;
end

if type==0
    Path = zeros(sum(TT)-length(TT)*L,K,'single');
    Xi = zeros(sum(TT)-length(TT)*(L+1),K,K,'single');
else
    Path = zeros(sum(TT)-length(TT)*L,1,'single');
    Xi = [];
end

for i = 1:N
    [X,XX,Y,Ti] = loadfile(Xin{i},T{i},hmm.train);
    XX_i = cell(1); XX_i{1} = XX;
    hmm_i = hmm;
    t = (1:(sum(Ti)-length(Ti)*hmm.train.maxorder)) + tacc;
    t2 = (1:(sum(Ti)-length(Ti)*(hmm.train.maxorder+1))) + tacc2;
    tacc = tacc + length(t); tacc2 = tacc2 + length(t2);
    if type==0
        data = struct('X',X,'C',NaN(sum(Ti)-length(Ti)*hmm.train.maxorder,K));
        [gamma,~,xi] = hsinference(data,Ti,hmm_i,Y,[],XX_i);
        checkGamma(gamma,Ti,hmm_i.train,i);
        Path(t,:) = single(gamma);
        Xi(t2,:,:) = xi;
    else
        BIGNbatch = hmm.train.BIGNbatch;
        hmm.train = rmfield(hmm.train,'BIGNbatch');
        pca = hmm.train.pca; hmm.train.pca = 0; 
        Path(t,:) = hmmdecode(X,Ti,hmm,type,Y);
        hmm.train.pca = pca; 
        hmm.train.BIGNbatch = BIGNbatch;
    end
    
end


end
