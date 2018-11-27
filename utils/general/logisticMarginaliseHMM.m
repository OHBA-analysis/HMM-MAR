function hmm_out = logisticMarginaliseHMM(hmm_in,iY)
% This function takes a multiple logistic strcuture HMM and returns the
% equivalent HMM with only the marginalised component iY of the
% logisticYdim components.

Xfulldim = hmm_in.train.ndim;
Xdim = Xfulldim - hmm_in.train.logisticYdim;
Xfulldim_new=Xdim+1;
select_vec = [1:Xdim,Xdim+iY];
hmm_out=hmm_in;
hmm_out.train.origlogisticYdim = hmm_in.train.logisticYdim;

%update S and options:
hmm_out.train.S=hmm_out.train.S(1:Xfulldim_new,1:Xfulldim_new);
hmm_out.train.Sind=hmm_out.train.Sind(1:Xfulldim_new,1:Xfulldim_new);
hmm_out.train.logisticYdim = 1;
hmm_out.train.ndim = Xfulldim_new;

%update W:
for st=1:hmm_in.train.K
    hmm_out.state(st).W.Mu_W = hmm_out.state(st).W.Mu_W(select_vec,select_vec);
    hmm_out.state(st).W.S_W = hmm_out.state(st).W.S_W(select_vec,select_vec,select_vec);
    hmm_out.state(st).W.iS_W = hmm_out.state(st).W.iS_W(select_vec,select_vec,select_vec);
end

%update alpha and sigma
for st=1:hmm_out.train.K
    %recall that for logistic setups, alpha has dimension Xdim x logisticYdim 
    hmm_out.state(st).alpha.Gam_rate = hmm_out.state(st).alpha.Gam_rate(:,iY);
    hmm_out.state(st).sigma.Gam_shape = hmm_out.state(st).sigma.Gam_shape(select_vec,select_vec);
    hmm_out.state(st).sigma.Gam_rate = hmm_out.state(st).sigma.Gam_rate(select_vec,select_vec);
end

%update prior over sigma
for st=1:hmm_out.train.K
    hmm_out.state(st).prior.sigma.Gam_shape = hmm_out.state(st).prior.sigma.Gam_shape(select_vec,select_vec);
    hmm_out.state(st).prior.sigma.Gam_rate = hmm_out.state(st).prior.sigma.Gam_rate(select_vec,select_vec);
end

%update psi:
if isfield(hmm_in,'psi')
    if size(hmm_in.psi,2)==hmm_in.train.logisticYdim
        %has psi been calculated specific to this variable:
        hmm_out.psi = hmm_in.psi(:,iY);
    else
        hmm_out = rmfield(hmm_out,'psi');
    end
end

end