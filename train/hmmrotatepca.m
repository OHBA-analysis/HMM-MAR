function hmm = hmmrotatepca(hmm,Gamma,XXGXX)

for k = 1:length(hmm.state)

    % Orthogonalize W to the standard PCA subspace
    W = hmm.state(k).W.Mu_W;
    %[W,~] = svd(hmm.state(k).W.Mu_W,'econ');
    
%     % Enforce a sign convention on the coefficients:
%     % the largest element in each column will have a positive sign.
    [~,maxind] = max(abs(W), [], 1);
    [d1, d2] = size(W);
    colsign = sign(W(maxind + (0:d1:(d2-1)*d1)));
    W = bsxfun(@times, W, colsign);
    hmm.state(k).W.Mu_W = W;
    
    % recompute covariance of W
    v = hmm.Omega.Gam_rate / hmm.Omega.Gam_shape;
    SW = XXGXX{k} * W / sum(Gamma(:,k));
    M = W'*W+v*eye(size(W,2));
    iS_W = v*eye(size(W,2))+M\W'*SW; S_W = inv(iS_W);
    for n = 1:size(hmm.state(k).W.iS_W,1)
        hmm.state(k).W.iS_W(n,:,:) = iS_W;
        hmm.state(k).W.S_W(n,:,:) = S_W;
    end
    
end

