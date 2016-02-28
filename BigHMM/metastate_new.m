function metastate = metastate_new(rate,shape,gram,m)
metastate = struct();
metastate.W = struct();
metastate.W.iS_W = gram;
metastate.W.S_W = inv(gram);
metastate.W.Mu_W = metastate.W.S_W * m;
if ~isempty(rate)
    metastate.Omega = struct();
    metastate.Omega.Gam_rate = rate;
    metastate.Omega.Gam_shape = shape;
    if isvector(rate)
        metastate.Omega.Gam_irate = 1 ./ metastate.Omega.Gam_rate;
    else
        metastate.Omega.Gam_irate = inv(metastate.Omega.Gam_rate);
    end
end
end