function hmm = loadhmm(hmm0,T,K,centroids,P,Pi,Dir2d_alpha,Dir_alpha,gram,prior)
hmm = hmm0; hmm = rmfield(hmm,'state');
for k = 1:K
    s = state_new(centroids(k)); 
    s.prior = hmm0.state(k).prior;
    if isfield(prior,'Mean')
        s.prior.Mean.iS = prior.Mean.iS';
    end
    if isfield(prior,'Omega')
        s.prior.Omega.Gam_rate = prior.Omega.Gam_rate;
        s.prior.Omega.Gam_shape = prior.Omega.Gam_shape;
    end
    hmm.state(k) = s;
    if hmm0.train.zeromean && hmm0.train.order==0
        hmm.state(k).W.Mu_W = [];
    end
end
hmm.K = K;
hmm.train.Sind = formindexes([],hmm.train.S);
hmm.train.Sind = [true(1,size(hmm.train.S,1)); hmm.train.Sind];
if isempty(P)
    hmm.Dir_alpha = ones(1,K);
    hmm.prior.Dir_alpha = ones(1,K);
    hmm.Pi = ones(1,K) / K;
    hmm.Dir2d_alpha = ones(K) + ...
        (hmm.train.DirichletDiag-1) * eye(K);
    hmm.prior.Dir2d_alpha = hmm.Dir2d_alpha;
    hmm.P = zeros(K);
    for k = 1:K
        hmm.P(k,:) = hmm.Dir2d_alpha(k,:) / ...
            sum(hmm.Dir2d_alpha(k,:));
    end
else
    hmm.Pi = Pi; hmm.P = P;
    hmm.Dir_alpha = Dir_alpha; hmm.Dir2d_alpha = Dir2d_alpha;
end
if ~isempty(gram)
    hmm.Omega = struct();
    hmm.Omega.Gam_rate = gram + prior.Omega.Gam_rate;
    hmm.Omega.Gam_irate = inv(hmm.Omega.Gam_rate);
    hmm.Omega.Gam_shape = sum(T) + prior.Omega.Gam_shape;
end
end


function state = state_new(centroid)
state = struct();
state.Omega.Gam_shape = centroid.shape;
state.Omega.Gam_rate = centroid.rate;
state.Omega.Gam_irate = centroid.irate;
if isfield(centroid,'m')
    state.W.Mu_W = centroid.m';
    state.W.S_W = centroid.Sm;
end
end