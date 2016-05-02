function metastate = metastate_new(rate,shape,gram,m,covtype,Sind)
% does basically the same than updateW and updateOmega, but approximately,
% and using the accumulated sufficient statistics

metastate = struct();
metastate.W = struct();
ndim = size(m,2); npred = size(gram,1);
fullcovmat = strcmp(covtype,'full') || strcmp(covtype,'uniquefull');
uniquecovmat = strcmp(covtype,'uniquediag') || strcmp(covtype,'uniquefull');
eps = 1e-8;

if uniquecovmat
    if fullcovmat
        prec = inv(rate / shape);
    else
        prec = shape ./ rate;
    end
else
    metastate.Omega = struct();
    metastate.Omega.Gam_rate = rate;
    metastate.Omega.Gam_shape = shape;
    if fullcovmat
        metastate.Omega.Gam_irate = inv(metastate.Omega.Gam_rate);
    else
        metastate.Omega.Gam_irate = 1 ./ metastate.Omega.Gam_rate;
    end
    prec = metastate.Omega.Gam_irate * shape;
end

if fullcovmat
    metastate.W.iS_W = kron(gram,prec) + eps*eye(ndim*npred);
    metastate.W.S_W = inv(metastate.W.iS_W);
    metastate.W.Mu_W = metastate.W.S_W * m;
else
    metastate.W.iS_W = zeros(ndim,npred,npred);
    metastate.W.S_W = zeros(ndim,npred,npred);
    metastate.W.Mu_W = zeros(npred,ndim);  
    for n = 1:ndim
        metastate.W.iS_W(n,Sind(:,n),Sind(:,n)) = gram(Sind(:,n),Sind(:,n)) + eps*eye(sum(Sind(:,n)));
        metastate.W.S_W(n,Sind(:,n),Sind(:,n)) = inv(permute(metastate.W.iS_W(n,Sind(:,n),Sind(:,n)),[2 3 1]));
        metastate.W.Mu_W(Sind(:,n),n) = ...
            permute(metastate.W.S_W(n,Sind(:,n),Sind(:,n)),[2 3 1]) * m(Sind(:,n),n);
    end
end

end