function state = state_snew(rate,shape,gram,m,covtype,Sind)
% does basically the same than updateW and updateOmega, but approximately,
% and using the accumulated sufficient statistics

state = struct();
state.W = struct();
ndim = size(m,2); npred = size(gram,1);
fullcovmat = strcmp(covtype,'full') || strcmp(covtype,'uniquefull');
uniquecovmat = strcmp(covtype,'uniquediag') || strcmp(covtype,'uniquefull');
eps = 1e-8;
if nargin<6, Sind = true(npred,ndim);
else Sind = (Sind==1);
end
if isempty(Sind), regressed = true(1,ndim);
else, regressed = sum(Sind,1)>0;
end

% Omega
if uniquecovmat
    if fullcovmat
        prec = inv(rate / shape);
    else
        prec = shape ./ rate;
    end
else
    state.Omega = struct();
    state.Omega.Gam_shape = shape;
    if fullcovmat
        state.Omega.Gam_rate = rate; 
        state.Omega.Gam_irate = inv(state.Omega.Gam_rate);
        prec = state.Omega.Gam_irate * shape;
    else
        state.Omega.Gam_rate = zeros(size(rate));
        state.Omega.Gam_rate(regressed) = rate(regressed);
    end
    
end

% W
if ~isempty(Sind)
    if fullcovmat
        mlW = gram \ m;
        Gram = kron(gram,prec);
        state.W.iS_W = Gram + eps*eye(ndim*npred);
        state.W.S_W = inv(state.W.iS_W);
        muW = state.W.S_W * Gram * mlW(:);
        state.W.Mu_W = reshape(muW,ndim,numel(muW)/ndim)';
    else
        state.W.iS_W = zeros(ndim,npred,npred);
        state.W.S_W = zeros(ndim,npred,npred);
        state.W.Mu_W = zeros(npred,ndim);
        for n = 1:ndim
            state.W.iS_W(n,Sind(:,n),Sind(:,n)) = gram(Sind(:,n),Sind(:,n)) + eps*eye(sum(Sind(:,n)));
            state.W.S_W(n,Sind(:,n),Sind(:,n)) = inv(permute(state.W.iS_W(n,Sind(:,n),Sind(:,n)),[2 3 1]));
            state.W.Mu_W(Sind(:,n),n) = ...
                permute(state.W.S_W(n,Sind(:,n),Sind(:,n)),[2 3 1]) * m(Sind(:,n),n);
        end
        if ndim==1
            state.W.iS_W = permute(state.W.iS_W,[2 3 1]);
            state.W.S_W = permute(state.W.S_W,[2 3 1]);
        end
    end
else
    state.W.S_W = [];
    state.W.Mu_W = [];
end

end
