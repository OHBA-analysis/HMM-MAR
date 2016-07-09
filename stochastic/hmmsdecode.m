function [Path,Xi] = hmmsdecode(Xin,T,metahmm,type,markovTrans)
% 1) Compute the stata time courses or viterbi paths 
% 2) Compute the entropy,avglifetime,nochanges from it
%
% INPUTS
% Xin: cell with strings referring to the subject files
% T: cell of vectors, where each element has the length of each trial per
% metahmm: the metastates computed from trainBigHMM
% type: 0, state time courses; 1, viterbi path
% markovTrans.P and Pi: transition prob table and initial prob
% NOTE: computations of stats now done in getshmmstats.m
%
% Diego Vidaurre, OHBA, University of Oxford (2015)

if nargin<4, type = 0; end
if nargin<5, markovTrans = []; end

for i = 1:length(T)
    if size(T{i},1)==1, T{i} = T{i}'; end
end

N = length(Xin);
K = length(metahmm.state);
TT = []; for i=1:N, TT = [TT; T{i}]; end
tacc = 0; tacc2 = 0; 

if type==0
    Path = zeros(sum(TT)-length(TT)*metahmm.train.order,K,'single');
    Xi = zeros(sum(TT)-length(TT)*(metahmm.train.order+1),K,K,'single');
else
    Path = zeros(sum(TT)-length(TT)*metahmm.train.order,1,'single');
    Xi = [];
end
     
BIGuniqueTrans = metahmm.train.BIGuniqueTrans;
if ~BIGuniqueTrans
    if isempty(markovTrans)
        error('Parameter markovTrans needs to be supplied')
    end
    P = markovTrans.P;
    Pi = markovTrans.Pi;
    Dir2d_alpha = markovTrans.Dir2d_alpha;
    Dir_alpha = markovTrans.Dir_alpha;
end

for i = 1:N
    [X,XX,Y,Ti] = loadfile(Xin{i},T{i},metahmm.train);    
    XX_i = cell(1); XX_i{1} = XX;
    if BIGuniqueTrans
        metahmm_i = metahmm;
    else
        metahmm_i = copyhmm(metahmm,P(:,:,i),Pi(:,i)',Dir2d_alpha(:,:,i),Dir_alpha(:,i)');
    end
    t = (1:(sum(Ti)-length(Ti)*metahmm.train.order)) + tacc;
    t2 = (1:(sum(Ti)-length(Ti)*(metahmm.train.order+1))) + tacc2;
    tacc = tacc + length(t); tacc2 = tacc2 + length(t2);
    if type==0
        data = struct('X',X,'C',NaN(sum(Ti)-length(Ti)*metahmm.train.order,K));
        [gamma,~,xi] = hsinference(data,Ti,metahmm_i,Y,[],XX_i);
        Path(t,:) = single(gamma);
        Xi(t2,:,:) = xi;
    else
        BIGNbatch = metahmm.train.BIGNbatch;
        metahmm.train = rmfield(metahmm.train,'BIGNbatch');
        Path(t,:) = hmmdecode(X,Ti,metahmm,type,Y);
        metahmm.train.BIGNbatch = BIGNbatch; 
    end

end


end
