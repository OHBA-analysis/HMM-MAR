function [acc_chance,p_cluster] = group_accuracy_significance(acc_true,acc_perms,nPerms,pthresh)
% This function implements the cluster significance tests as outlined by
% Stelzer 2013; unlike int he original publication this is inferred over
% time rather than space. This allows for inference of temporal clusters in
% a 2D temporal generalisation method, or 1D clusters on a single temporal
% axis.
%
% INPUTS:
% 
% acc_true      The true mean accuracy over subjects. This should be of
%               dimension [T1 x T2 x P], where T1 and T2 are lengths of
%               time (this for time generalisation plots; if inferring
%               clusters in 1D time, then T2 should be 1); P is the number
%               of regressor dimensions.
% acc_perms     The set of permuted label accuracy data sets. These should
%               be of dimension [T1 x T2 x P x L x N], where L is the
%               number of label permutations applied and N is the number of
%               subjects.
% nPerms        The number of bootstrap permutations to compute (default 10000)
% pThresh       The chance decoding accuracy (this is used both for the
%               output acc_chance and also the cluster threshold definition
%               - default is 0.001)
%
% OUTPUTS:
% acc_chance    A vector of dimension [T1 x T2 x P] showing the non
%               parameteric chance decode accuracy (using p = pthresh)
% p_cluster     For the data in acc_true, the probability of observing a
%               cluster of this size by chance.
%
%
% Reference: ?Stelzer et al 2013, Neuroimage: Statistical inference and multiple testing correction in 
% classification-based multi-voxel pattern analysis (MVPA): Random permutations and cluster size control
%
% Software Author: Cam Higgins, OHBA, University of Oxford  



[NT,NF,NR] = size(acc_true); % note that clusters are defined over dimensions NT and NF
[NTp,NFp,NRp,NPp,NSj] = size(acc_perms);
if NT ~= NTp || NF~=NFp || NR~=NRp
    error('Dimensions of group mean and permutations do not match');
end
if nargin<3 || isempty(nPerms)
    nPerms = 10000;
end
if nargin<4 || isempty(pthresh)
    pthresh = 0.001;
end
con = 6; % this corresponds to a cluster connectivity criterion of connected faces (ie clusters do not include diagonally connected points)

% step 1: generate null distribution of group means by bootstrap
% aggregation:
for i = 1:nPerms
    permselect = false(NPp,NSj);
    permselect(randi(NPp,NSj,1) + [0:NSj-1]'*NPp) = true;
    acc_bootstrap(:,:,:,i) = mean(acc_perms(:,:,:,permselect),4);
end

% step 2: find chance accuracy level at threshold given:
acc_chance = prctile(acc_bootstrap,(1-pthresh)*100,4);

% step 3: compute statistics of clusters above acc_chance:
for i=1:nPerms
    superthresh = acc_bootstrap(:,:,:,i)>acc_chance;
    for iR=1:NR
        [imlabel LL] = spm_bwlabel(double(superthresh(:,:,iR)),con);
        tmp = unique(imlabel);
        tmp(tmp==0) = [];
        nL = 0;
        if ~isempty(tmp)
            for kk = 1:numel(tmp); %loop over clusters
                k = tmp(kk);
                nL(k) = sum(sum(imlabel==k));
            end
        end
        nulldist(iR,i)=max(nL);
    end
end
    
superthresh_true = acc_true>acc_chance;
p_cluster = zeros(size(acc_true));
for iR = 1:NR
    [imlabel LL] = spm_bwlabel(double(superthresh_true(:,:,iR)),con);
    tmp = unique(imlabel);
    tmp(tmp==0) = [];
    cpimlabel = zeros(size(imlabel));
    if ~isempty(tmp)
        for kk = 1:numel(tmp) %loop over clusters
          k = tmp(kk);
          nL = sum(sum(imlabel==k));
          cp = mean(nL>nulldist(iR,:));
          cpimlabel(imlabel==k)=cp;
        end
    end
    p_cluster(:,:,iR) = cpimlabel;
end


end