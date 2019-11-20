function hmm = hmmperturb(hmm,epsilon)

if nargin<2, epsilon = 1; end
S = hmm.train.S==1; regressed = sum(S,1)>0;
p = hmm.train.lowrank;

K = length(hmm.state);
for k = 1:K
    setstateoptions
    if p > 0 % pcaHMM
        for n = 1:size(hmm.state(k).W.S_W,2)
            Cov = permute(hmm.state(k).W.S_W(n,:,:),[2 3 1]);
            Cov = Cov + 0.001 * eye(length(Cov));
            Cov = (Cov' + Cov)/2; 
            hmm.state(k).W.Mu_W(n,:) = hmm.state(k).W.Mu_W(n,:) + ...
                epsilon * mvnrnd(hmm.state(k).W.Mu_W(n,:),Cov);
        end
    elseif isfield(hmm.state(k),'W') && ~isempty(hmm.state(k).W.Mu_W)
        if length(size(hmm.state(k).W.S_W)) == 3 || ...
                size(hmm.state(k).W.S_W,2) == 1 % diagonal covmat
            for n = 1:size(hmm.state(k).W.S_W,1)
                if ~regressed(n), continue; end
                Cov = permute(hmm.state(k).W.S_W(n,Sind(:,n),Sind(:,n)),[2 3 1]);
                Cov = Cov + 0.001 * eye(length(Cov));
                hmm.state(k).W.Mu_W(Sind(:,n),n) = hmm.state(k).W.Mu_W(Sind(:,n),n) + ...
                    epsilon * mvnrnd(hmm.state(k).W.Mu_W(Sind(:,n),n),Cov)';
            end
        else % full covmat
            mu = hmm.state(k).W.Mu_W(:);
            Cov = hmm.state(k).W.S_W;
            Cov = Cov + 0.001 * eye(length(Cov));
            W = mvnrnd(mu,Cov)';
            hmm.state(k).W.Mu_W =  hmm.state(k).W.Mu_W + ...
                epsilon * reshape(W,size(hmm.state(k).W.Mu_W));
        end
    else % if there is no regression coeff / mean , perturbe the noise cov mat
        if size(hmm.state(k).Omega.Gam_rate,1) == 1 % diagonal
            r = randn(size(hmm.state(k).Omega.Gam_rate));
            hmm.state(k).Omega.Gam_rate = hmm.state(k).Omega.Gam_rate + epsilon * 0.01 * r.^2;
        else
            ndim = length(hmm.state(k).Omega.Gam_rate);
            r = randn(ndim,1);
            hmm.state(k).Omega.Gam_rate = hmm.state(k).Omega.Gam_rate + epsilon * 0.01 * (r' * r);
        end
    end
    
end

end