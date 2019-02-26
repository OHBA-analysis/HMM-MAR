function D = hmm_kl (hmm_p,hmm_q)
% Computes Kullback-Leibler divergence between two Hidden Markov Model
% distributions, through an approximation (an upper bound) as proposed in
% M. Do (2003). IEEE Signal Processing Letters 10
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2018)

K = length(hmm_p.Pi);
if K~=length(hmm_q.Pi)
    error(['The two HMMs must have the same number of states, ' ...
        'and their order must correspond'])
end
if (hmm_p.train.order ~= hmm_q.train.order) || ...
        (~strcmpi(hmm_p.train.covtype,hmm_q.train.covtype)) || ...
        (length(hmm_p.train.embeddedlags) ~= length(hmm_q.train.embeddedlags)) || ...
        (any(hmm_p.train.embeddedlags ~= hmm_q.train.embeddedlags)) || ...
        (hmm_p.train.zeromean ~= hmm_q.train.zeromean)  
   error('The state configuration of the two HMMs must be identical') 
end
hmm = hmm_p; setstateoptions;
if isfield(hmm_p.state(1),'W')
    ndim = size(hmm_p.state(1).W.Mu_W,2);
else
    ndim = size(hmm_p.state(1).Omega.Gam_rate,2);
end
S = hmm.train.S==1;
regressed = sum(S,1)>0;

D = 0;
nu = compute_nu (hmm_p.Pi,hmm_p.P); % weight vector

% Non-state specific stuff
if strcmp(hmm.train.covtype,'uniquediag')
    for n = 1:ndim
        if ~regressed(n), continue; end
        D = D + gamma_kl(hmm.Omega.Gam_shape,hmm.prior.Omega.Gam_shape, ...
            hmm.Omega.Gam_rate(n),hmm.prior.Omega.Gam_rate(n));
    end
elseif strcmp(hmm.train.covtype,'uniquefull')
    D = D + wishart_kl(hmm.Omega.Gam_rate(regressed,regressed),...
        hmm.prior.Omega.Gam_rate(regressed,regressed), ...
        hmm.Omega.Gam_shape,hmm.prior.Omega.Gam_shape);
end

% State specific stuff
for k = 1:K
    
    % Trans probabilities
    kk = hmm.train.Pstructure(k,:);
    D = D + nu(k) * dirichlet_kl(hmm_p.Dir2d_alpha(k,kk),hmm_q.prior.Dir2d_alpha(k,kk));
    
    % State distribution
    hs = hmm_p.state(k);
    hs0 = hmm_q.state(k);
    
    if ~isempty(hs.W.Mu_W)
        if train.uniqueAR || ndim==1
            if train.uniqueAR || ndim==1
                D = D + nu(k) * gauss_kl(hs.W.Mu_W, hs0.W.Mu_W, hs.W.S_W, hs0.W.S_W);
            else
                D = D + nu(k) * gauss_kl(hs.W.Mu_W, hs0.W.Mu_W, ...
                    permute(hs.W.S_W,[2 3 1]), permute(hs0.W.S_W,[2 3 1]));
            end
        elseif strcmp(train.covtype,'diag') || strcmp(train.covtype,'uniquediag')
            for n=1:ndim
                D = D + nu(k) * gauss_kl(hs.W.Mu_W(Sind(:,n),n),hs0.W.Mu_W(Sind(:,n),n), ...
                    permute(hs.W.S_W(n,Sind(:,n),Sind(:,n)),[2 3 1]),...
                    permute(hs0.W.S_W(n,Sind(:,n),Sind(:,n)),[2 3 1]));
            end
        else % full or uniquefull
            mu_w = hs.W.Mu_W';
            mu_w = mu_w(:);
            mu_w0 = hs0.W.Mu_W';
            mu_w0 = mu_w0(:);
            D = D + nu(k) * gauss_kl(mu_w,mu_w0, hs.W.S_W, hs0.W.S_W);
        end
    end
    
    switch train.covtype
        case 'diag'
            for n=1:ndim
                if ~regressed(n), continue; end
                D = D + nu(k) * gamma_kl(hs.Omega.Gam_shape,hs0.Omega.Gam_shape, ...
                    hs.Omega.Gam_rate(n),hs0.Omega.Gam_rate(n));
            end
        case 'full'
            try
                D = D + nu(k) * wishart_kl(hs.Omega.Gam_rate(regressed,regressed),...
                    hs0.Omega.Gam_rate(regressed,regressed), ...
                    hs.Omega.Gam_shape,hs0.Omega.Gam_shape);
            catch
                error(['Error computing kullback-leibler divergence of the cov matrix - ' ...
                    'Something strange with the data?'])
            end
    end
    
    if ~isempty(orders) && ~train.uniqueAR && ndim>1
        for n1=1:ndim
            for n2=1:ndim
                if (train.symmetricprior && n2<n1) || S(n1,n2)==0, continue; end
                D = D + nu(k) * gamma_kl(hs.sigma.Gam_shape(n1,n2),hs0.sigma.Gam_shape(n1,n2), ...
                    hs.sigma.Gam_rate(n1,n2),hs0.sigma.Gam_rate(n1,n2));
            end
        end
    end
    if ~isempty(orders)
        for i=1:length(orders)
            D = D + nu(k) * gamma_kl(hs.alpha.Gam_shape,hs0.alpha.Gam_shape, ...
                hs.alpha.Gam_rate(i),hs0.alpha.Gam_rate(i));
        end
    end
end

end

function nu = compute_nu (Pi,P)
eps = 1e-6;
nu = Pi * P;
while true
    nu0 = nu; 
    nu = nu * P;
    if mean(nu(:)-nu0(:))<eps, break; end
end  
end
