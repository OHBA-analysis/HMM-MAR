function hmm = hmmperturb(hmm,epsilon)

if nargin<2, epsilon = 1; end

K = length(hmm.state);
for k = 1:K
    setstateoptions
    if isfield(hmm.state(k),'W') && ~isempty(hmm.state(k).W.Mu_W)
        if length(size(hmm.state(k).W.S_W)) == 3
           for n = 1:size(hmm.state(k).W.S_W,1)
               Cov = permute(hmm.state(k).W.S_W(n,Sind(:,n),Sind(:,n)),[2 3 1]);
               hmm.state(k).W.Mu_W(Sind(:,n),n) = hmm.state(k).W.Mu_W(Sind(:,n),n) + ...
                   epsilon * mvnrnd(hmm.state(k).W.Mu_W(:,n),Cov)';
           end
        else
            mu = hmm.state(k).W.Mu_W(:);
            Cov = hmm.state(k).W.S_W;
            W = mvnrnd(mu,Cov)';
            hmm.state(k).W.Mu_W =  hmm.state(k).W.Mu_W + ... 
                epsilon * reshape(W,size(hmm.state(k).W.Mu_W));
        end
    else % if there is no regression coeff / mean , perturbe the noise cov mat
        if size(hmm.state(k).Omega.Gam_rate,1) == 1 % diagonal
            hmm.state(k).Omega.Gam_rate = hmm.state(k).Omega.Gam_rate + ...
                epsilon * 0.01 * randn(size(hmm.state(k).Omega.Gam_rate)) .*  ...
                hmm.state(k).Omega.Gam_rate; 
            hmm.state(k).Omega.Gam_rate = max(hmm.state(k).Omega.Gam_rate,0.0001);
        else
            ndim = length(hmm.state(k).Omega.Gam_rate);
            for n = 1:ndim
                hmm.state(k).Omega.Gam_rate(n,n) = ...
                    epsilon * 0.01 * randn(1) .* hmm.state(k).Omega.Gam_rate(n,n);
                hmm.state(k).Omega.Gam_rate(n,n) = max(hmm.state(k).Omega.Gam_rate(n,n),0.0001);
            end
        end
    end

end



end