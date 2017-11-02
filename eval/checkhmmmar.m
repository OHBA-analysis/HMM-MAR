function isstable = checkhmmmar(hmm)
% Checks for stability, which implies stationarity  
% Needs the Econometrics Toolbox
%
% INPUTS:
%
% hmm           hmm structure  
%
% OUTPUTS
% stability     vector indicating stability for each MAR (K x 1)  
%
% Author: Diego Vidaurre, OHBA, University of Oxford

K = length(hmm.state); ndim = size(hmm.state(1).W.Mu_W,2);
if hmm.train.order==0, error('This is a HMM-Gaussian model, not a HMM-MAR \n'); end
isstable = zeros(K,1);

for k = 1:K
    setstateoptions;
    order = orders(end);
    W = {};
    for j=1:order
       W{j} = zeros(ndim,ndim); 
    end
    
    for j=1:length(orders)
        o = orders(j);
        W{o} = hmm.state(k).W.Mu_W((1:ndim) + ndim * (j-1),:);
    end
    mar = vgxset('AR',W);
    isstable(k) = vgxqual(mar);
end

end