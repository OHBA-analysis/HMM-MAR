function hmm = updateW_addHMM(hmm,Gamma,residuals,XX,Tfactor)
% assumes covtype==uniquediag (otherwise the matrix multiplicatin becomes huge)

if nargin < 5, Tfactor = 1; end
setstateoptions;
regressed = sum(S,1)>0;

K = size(Gamma,2);
Gamma = [Gamma (size(Gamma,2)-sum(Gamma,2)) ];
np = size(XX,2); ndim = size(residuals,2); 

X = zeros(size(XX,1),np * (K+1));
for k = 1:K+1
   X(:,(1:np) + (k-1)*np) = bsxfun(@times, XX, Gamma(:,k)); 
end
gram = X' * X; 

hmm.state_shared = struct();

for n = 1:ndim
    
    if ~regressed(n), continue; end
    
    Regterm = zeros(np*(K+1),1); Sind_all = []; 
    for k = 1:K+1
        ndim_n = sum(S(:,n)>0);
        if ndim_n==0 && train.zeromean==1, continue; end
        Sind_all = [Sind_all; Sind(:,n)];
        regterm = []; I = [];
        if ~train.zeromean, regterm = hmm.state(k).prior.Mean.iS(n); I = true; end
        if hmm.train.order > 1
            alphaterm = ...
                repmat( (hmm.state(k).alpha.Gam_shape ./  hmm.state(k).alpha.Gam_rate), ndim_n, 1);
            if np>1
                regterm = [regterm; ...
                    repmat(hmm.state(k).sigma.Gam_shape(S(:,n),n) ./ ...
                    hmm.state(k).sigma.Gam_rate(S(:,n),n), length(orders), 1) .* ...
                    alphaterm(:) ];
            else
                regterm = [regterm; alphaterm(:)];
            end
            I = [I; S(:,n)];
        end
        ind = (1:np) + (k-1)*np;
        Regterm(ind(I)) = regterm;
    end
    Sind_all = Sind_all == 1; 
    c = hmm.Omega.Gam_shape / hmm.Omega.Gam_rate(n);
    Regterm = diag(Regterm);

    iS_W = Regterm + Tfactor * c * gram(Sind_all,Sind_all);
    iS_W = (iS_W + iS_W') / 2; 
    S_W = inv(iS_W);
    Mu_W = Tfactor * c * S_W * X(:,Sind_all)' * residuals(:,n);

    hmm.state_shared(n).iS_W = zeros(size(gram));
    hmm.state_shared(n).S_W = zeros(size(gram));
    hmm.state_shared(n).Mu_W = zeros(size(X,2),1);
    
    hmm.state_shared(n).iS_W(Sind_all,Sind_all) = iS_W;
    hmm.state_shared(n).S_W(Sind_all,Sind_all) = S_W;
    hmm.state_shared(n).Mu_W(Sind_all) = Mu_W;
      
end

% copy parameters into the state containers
for k = 1:K+1
    for n = 1:ndim
        ind = (1:np) + (k-1)*np;
        hmm.state(k).W.Mu_W(:,n) = hmm.state_shared(n).Mu_W(ind);
        hmm.state(k).W.iS_W(n,:,:) = hmm.state_shared(n).iS_W(ind,ind);
        hmm.state(k).W.S_W(n,:,:) = hmm.state_shared(n).S_W(ind,ind);
    end
end

% ML prediction
% B = pinv(Gamma) * residuals;
% for k = 1:length(hmm.state)
%     hmm.state(k).W.Mu_W = B(k,:);
% end

end
