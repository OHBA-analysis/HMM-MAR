function ness = updateW_ness(ness,Gamma,residuals,XX,Tfactor)
% assumes covtype==uniquediag (otherwise the matrix multiplication becomes huge)

if nargin < 5, Tfactor = 1; end
setstateoptions;
regressed = sum(S,1)>0;

K = size(Gamma,2); np = size(XX,2); ndim = length(ness.state_shared); 
Gamma = [Gamma (K-sum(Gamma,2)) ];
X = zeros(size(XX,1),np * (K+1));
for k = 1:K+1
   X(:,(1:np) + (k-1)*np) = bsxfun(@times, XX, Gamma(:,k)); 
end
gram = X' * X; %repmat(XX,1,K+1); 

ness.state_shared = struct();

for n = 1:ndim
    
    if ~regressed(n), continue; end
    
    Regterm = zeros(np*(K+1),1); Sind_all = []; 
    for k = 1:K+1
        ndim_n = sum(S(:,n)>0);
        %if ndim_n==0 && train.zeromean==1, continue; end
        Sind_all = [Sind_all; Sind(:,n)];
        regterm = []; I = [];
        if ~train.zeromean, regterm = ness.state(k).prior.Mean.iS(n); I = true; end
        if ness.train.order > 0
            alphaterm = ...
                repmat( (ness.state(k).alpha.Gam_shape ./  ...
                ness.state(k).alpha.Gam_rate), ndim_n, 1);
            sigmaterm = repmat(ness.state(k).sigma.Gam_shape(S(:,n),n) ./ ...
                ness.state(k).sigma.Gam_rate(S(:,n),n), length(orders), 1);
            regterm = [regterm; sigmaterm .* alphaterm(:) ];
            I = [I; Sind(:,n)];
        end
        I = find(I);
        ind = (1:np) + (k-1)*np;
        Regterm(ind(I)) = regterm;
    end
    Sind_all = Sind_all == 1; 
    c = ness.Omega.Gam_shape / ness.Omega.Gam_rate(n);
    Regterm = diag(Regterm);

    iS_W = Regterm + Tfactor * c * gram(Sind_all,Sind_all);
    iS_W = (iS_W + iS_W') / 2; 
    S_W = inv(iS_W);
    Mu_W = Tfactor * c * S_W * X(:,Sind_all)' * residuals(:,n);

    ness.state_shared(n).iS_W = zeros(size(gram));
    ness.state_shared(n).S_W = zeros(size(gram));
    ness.state_shared(n).Mu_W = zeros(size(X,2),1);
    
    ness.state_shared(n).iS_W(Sind_all,Sind_all) = iS_W;
    ness.state_shared(n).S_W(Sind_all,Sind_all) = S_W;
    ness.state_shared(n).Mu_W(Sind_all) = Mu_W;
      
end

% copy parameters into the state containers
for k = 1:K+1
    for n = 1:ndim
        ind = (1:np) + (k-1)*np;
        ness.state(k).W.Mu_W(:,n) = ness.state_shared(n).Mu_W(ind);
        ness.state(k).W.iS_W(n,:,:) = ness.state_shared(n).iS_W(ind,ind);
        ness.state(k).W.S_W(n,:,:) = ness.state_shared(n).S_W(ind,ind);
    end
end

end
