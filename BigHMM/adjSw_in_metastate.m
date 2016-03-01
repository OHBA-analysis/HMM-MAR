function metastates = adjSw_in_metastate(metastates)
% adjust the dimensions of W.S_W
if ~isfield(metastates.state(1),'W') || isempty(metastates.state(1).W.Mu_W)
    return
end
[nprec,ndim] = size(metastates.state(1).W.Mu_W);
for k = 1:length(metastates.state)
    diag = (isfield(metastates,'Omega') && isvector(metastates.Omega.Gam_rate)) || ...
        (isfield(metastates.state(k),'Omega') && isvector(metastates.state(k).Omega.Gam_rate));
    iS_W = metastates.state(k).W.iS_W;
    S_W = metastates.state(k).W.S_W;
    if diag
        metastates.state(k).W.S_W = zeros(ndim,nprec,nprec);
        metastates.state(k).W.iS_W = zeros(ndim,nprec,nprec);
        for n=1:ndim
            metastates.state(k).W.S_W(n,:,:) = S_W;
            metastates.state(k).W.iS_W(n,:,:) = iS_W;
        end
    else
        metastates.state(k).W.S_W = zeros(ndim*nprec,ndim*nprec);
        for n = 1:ndim
            ind = (1:nprec) + (n-1)*nprec;
            metastates.state(k).W.S_W(ind,ind) = S_W;
            metastates.state(k).W.iS_W(ind,ind) = iS_W;
        end
    end
end
end