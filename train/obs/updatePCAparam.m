function hmm = updatePCAparam (hmm,Gammasum,XXGXX,Tfactor)

K = length(hmm.state); ndim = hmm.train.ndim;
if nargin<5, Tfactor = 1; end
p = hmm.train.lowrank; 

for k = 1:K
    
    % unlike Bishop's mixture of PCA, we don't have a mean vector per state here
    v = hmm.Omega.Gam_rate / hmm.Omega.Gam_shape;
    W = hmm.state(k).W.Mu_W; % posterior dist of the precision matrix
    S = XXGXX{k};% / Gammasum(k);
    %regterm = diag(hmm.state(k).beta.Gam_shape ./ hmm.state(k).beta.Gam_rate);
    SW = S * W;
    M = W'*W + v*eye(p);
    
    % W
    iS_W = v*eye(p) + M\W'*SW / Gammasum(k); 
    S_W = inv(iS_W);
    hmm.state(k).W.Mu_W = SW * S_W / Gammasum(k);
    for n = 1:ndim
        hmm.state(k).W.iS_W(n,:,:) = iS_W;
        hmm.state(k).W.S_W(n,:,:) = S_W;
    end

    % Omega
    Wnew = hmm.state(k).W.Mu_W;
    omega_i = mean(diag(S - SW * (M \ Wnew')));
    % replace
    hmm.Omega.Gam_rate = hmm.Omega.Gam_rate - hmm.Omega.Gam_rate_state(k);
    hmm.Omega.Gam_shape = hmm.Omega.Gam_shape - hmm.Omega.Gam_shape_state(k);
    hmm.Omega.Gam_rate_state(k) = 0.5 * Tfactor * omega_i;
    hmm.Omega.Gam_shape_state(k) = 0.5  * Tfactor * Gammasum(k);
    hmm.Omega.Gam_rate = hmm.Omega.Gam_rate + hmm.Omega.Gam_rate_state(k);
    hmm.Omega.Gam_shape = hmm.Omega.Gam_shape + hmm.Omega.Gam_shape_state(k);
    
    % prior beta
    %hmm.state(k).beta.Gam_shape = hmm.state(k).prior.beta.Gam_shape + 0.5 * ndim;
    %hmm.state(k).beta.Gam_rate = hmm.state(k).prior.beta.Gam_rate + ...
    %    sum(hmm.state(k).W.Mu_W.^2) + ...
    %    diag(permute(sum(hmm.state(k).W.S_W,1),[2 3 1]))';
    
end

end