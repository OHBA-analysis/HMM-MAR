function hmm_out = logisticMergeHMM(hmm_marg,hmm_full,iY)
% This function takes a multiple logistic strcuture HMM and returns the
% equivalent HMM with only the marginalised component iY of the
% logisticYdim components.

Xfulldim = hmm_full.train.ndim;
Xdim = Xfulldim - hmm_full.train.logisticYdim;
Xfulldim_new=Xdim+1;
select_vec = [1:Xdim,Xdim+iY];
hmm_out=hmm_full;  

% %update S and options:
% hmm_out.train.S=hmm_out.train.S(1:Xfulldim_new,1:Xfulldim_new);
% hmm_out.train.Sind=hmm_out.train.Sind(1:Xfulldim_new,1:Xfulldim_new);
% hmm_out.train.logisticYdim = 1;
% hmm_out.train.ndim = Xfulldim_new;

%update W:
for st=1:hmm_full.train.K
    hmm_out.state(st).W.Mu_W(1:Xdim,Xdim+iY) = hmm_marg.state(st).W.Mu_W(1:Xdim,Xdim+1);
    hmm_out.state(st).W.S_W(Xdim+iY,1:Xdim,1:Xdim) = hmm_marg.state(st).W.S_W(Xdim+1,1:Xdim,1:Xdim);
    hmm_out.state(st).W.iS_W(Xdim+iY,1:Xdim,1:Xdim) = hmm_marg.state(st).W.iS_W(Xdim+1,1:Xdim,1:Xdim);
end

%update alpha and sigma
for st=1:hmm_out.train.K
    %recall that for logistic setups, alpha has dimension Xdim x logisticYdim
    hmm_out.state(st).alpha.Gam_rate(1:Xdim,iY) = hmm_marg.state(st).alpha.Gam_rate;
    hmm_out.state(st).sigma.Gam_shape(1:Xdim,Xdim+iY) = hmm_marg.state(st).sigma.Gam_shape(1:Xdim,Xdim+1);
    hmm_out.state(st).sigma.Gam_rate(1:Xdim,Xdim+iY) = hmm_marg.state(st).sigma.Gam_rate(1:Xdim,Xdim+1);
end

%update prior over sigma
for st=1:hmm_out.train.K
    hmm_out.state(st).prior.sigma.Gam_shape(1:Xdim,Xdim+iY) = hmm_marg.state(st).prior.sigma.Gam_shape(1:Xdim,Xdim+1);
    hmm_out.state(st).prior.sigma.Gam_rate(1:Xdim,Xdim+iY) = hmm_marg.state(st).prior.sigma.Gam_rate(1:Xdim,Xdim+1);
end

%update psi:
hmm_out.psi(:,iY) = hmm_marg.psi;

end