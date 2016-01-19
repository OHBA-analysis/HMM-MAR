function [centroids,markovTrans,subj_stats,prior,fehist,BIGbase_tmpfolder] = ...
    trainBigHMM(files,T,options)
% 1) Initialize BigHMM by estimating a separate HMM on each file,
%    and then using an adapted version of kmeans to clusterize the states
%    (see initBigHMM)
% 2) Run the BigHMM algorithm
%
% INPUTS
% files: cell with strings referring to the subject files
% T: cell of vectors, where each element has the length of each trial per
% options: HMM options for both the subject and the group runs
%
% COVTYPE=UNIQUEFULL/UNIQUEDIAG NOT COMPLETELY IMPLEMENTED!!
% Diego Vidaurre, OHBA, University of Oxford (2015)

N = length(files);

if ~isfield(options,'K'), error('K was not specified');
else K = options.K; end
% Specific BigHMM options
if ~isfield(options,'BIGNbatch'), BIGNbatch = 10;
else BIGNbatch = options.BIGNbatch; end
if ~isfield(options,'BIGcyc'), BIGcyc = 200;
else BIGcyc = options.BIGcyc; end
if ~isfield(options,'BIGmincyc'), BIGmincyc = 50;
else BIGmincyc = options.BIGmincyc; end
if ~isfield(options,'BIGundertol_tostop'), BIGundertol_tostop = 5;
else BIGundertol_tostop = options.BIGundertol_tostop; end
if ~isfield(options,'BIGtol'), BIGtol = 1e-5;
else BIGtol = options.BIGtol; end
if ~isfield(options,'BIGinitcyc'), BIGinitcyc = 4;
else BIGinitcyc = options.BIGinitcyc; end
if ~isfield(options,'BIGforgetrate'), BIGforgetrate = 4;
else BIGforgetrate = options.BIGforgetrate; end
if ~isfield(options,'BIGdelay'), BIGdelay = 1;
else BIGdelay = options.BIGdelay; end
if ~isfield(options,'BIGbase_weights'), BIGbase_weights = 1; % smaller will promote democracy
else BIGbase_weights = options.BIGbase_weights; end
if ~isfield(options,'BIGbase_tmpfolder'), BIGbase_tmpfolder = tempname; % smaller will promote democracy
else BIGbase_tmpfolder = options.BIGbase_tmpfolder; end
system(['mkdir ',BIGbase_tmpfolder]);

% HMM-MAR options
if ~isfield(options,'zeromean'), options.zeromean = 0; end
if ~isfield(options,'covtype'), options.covtype = 'full'; end
if ~isfield(options,'cyc'), options.cyc = 50; end
if ~isfield(options,'initcyc'), options.initcyc = 50; end
if ~isfield(options,'initrep'), options.initrep = 3; end
options.verbose = 0; % shut up the individual hmmmar output

if strcmp(options.covtype,'full') || strcmp(options.covtype,'diag')
    BIGcovtype = 'subject';
else
    if options.zeromean==1
        error('Neither states'' mean nor covariance are allow to vary')
    end
    BIGcovtype = 'unique';
end
    
if isfield(options,'order') && options.order>0, error('BigHMM only implemented for order=0'); 
else options.order=0; end

X = loadfile(files{1}); ndim = size(X,2); sumT = 0;
subjfe_init = zeros(N,3);
loglik_init = zeros(N,1);
statekl_init = zeros(K,1);
subjfe = zeros(N,3,BIGcyc);
loglik = zeros(N,BIGcyc);
statekl = zeros(K,BIGcyc);

% Initialization
if ~isfield(options,'centroids')
    
    subj_gamma_init = zeros(N,K);
    subj_m_init = zeros(ndim,N,K);
    if strcmp(options.covtype,'diag')
        subj_gram_init = zeros(ndim,N,K); gram_init = [];
    elseif strcmp(options.covtype,'full')
        subj_gram_init = zeros(ndim,ndim,N,K); gram_init = [];
    elseif strcmp(options.covtype,'uniquediag')
        gram_init = zeros(1,ndim); subj_gram_init = [];
    else % uniquefull
        gram_init = zeros(ndim,ndim); subj_gram_init = [];
    end
    centroids_init = struct('m',[],'Sm',[],'rate',[],'shape',[],'irate',[],'type',[]);
    P_init = cell(N,1); Pi_init = cell(N,1);
    Dir2d_alpha_init = cell(N,1); Dir_alpha_init = cell(N,1);

    best_fe = Inf;
    for cycle = 1:BIGinitcyc
        
        % train individual HMMs
        I = randperm(N);
        for ii = 1:length(I)
            i = I(ii);
            X = loadfile(files{i});
            sumT = sumT + sum(T{i});
            if cycle==1
                if ii==1
                    range_data = range(X);
                else
                    range_data = max(range_data,range(X));
                end
            end
            % Running the individual HMM
            [hmm_i,Gam] = hmmmar(X,T{i},options); Gam = single(Gam);
            %Gam = single([Gam zeros(size(Gam,1),K-size(Gam,2))]);
            save(strcat(BIGbase_tmpfolder,'/Gamma',num2str(i),'.mat'),'Gam')
            fprintf('Init run %d, subject %d \n',cycle,ii);
            P_init{i} = hmm_i.P; Pi_init{i} = hmm_i.Pi;
            Dir2d_alpha_init{i} = hmm_i.Dir2d_alpha; Dir_alpha_init{i} = hmm_i.Dir_alpha;
            K_i = length(hmm_i.state);
            if options.zeromean==1
                for j = 1:K_i
                    hmm_i.state(j).W.Mu_W = zeros(1,ndim);
                end
            end
            % Reassigning ordering of the states according the closest centroids
            if ii==1
                assig = 1:K_i;
            else
                dist = Inf(K_i,K);
                for j = 1:K_i
                    if strcmp(BIGcovtype,'subject')
                        rate = hmm_i.state(j).Omega.Gam_rate;
                        shape = hmm_i.state(j).Omega.Gam_shape;
                        s = centroid_new(rate, shape, hmm_i.state(j).W.Mu_W',ones(1,ndim)./sum(Gam(:,j)),BIGcovtype);
                        for k = 1:K
                            dist(j,k) = symm_kl_div(s, centroids_init(k));
                        end
                    else
                        for k = 1:K
                            dist(j,k) = sqrt(sum( (hmm_i.state(j).W.Mu_W' - centroids_init(k).m).^2));
                        end
                    end
                end
                assig = munkres(dist); % linear assignment problem
            end
            % update individual statistics
            for k=1:K_i,
                s = hmm_i.state(k);
                subj_m_init(:,i,assig(k)) = s.W.Mu_W';
                subj_gamma_init(i,assig(k)) = sum(Gam(:,k));
                % cov mats: note that these are the individual ones, and,
                % hence, an underestimation of the group ones
                if strcmp(options.covtype,'full') 
                    subj_gram_init(:,:,i,assig(k)) = s.Omega.Gam_rate;
                elseif strcmp(options.covtype,'diag')
                    subj_gram_init(:,i,assig(k)) = s.Omega.Gam_rate;
                end
            end
            if strcmp(options.covtype,'uniquefull') || strcmp(options.covtype,'uniquediag')
                gram_init = gram_init + hmm.Omega.Gam_rate;
            end
            % updating the centroids
            for k = 1:K
                gc = sum(subj_gamma_init(I(1:ii),k));
                if strcmp(options.covtype,'full')
                    prec = gc * (ndim+0.1-1) * diag(1./range_data);
                    Smc = inv(prec + diag(1 ./ ((range_data/2).^2) ));
                    mc = Smc * prec * sum(subj_m_init(:,I(1:ii),k),2) / gc;
                    cc = sum(subj_gram_init(:,:,I(1:ii),k),3);
                    centroids_init(k) = centroid_new(cc,gc,mc,Smc,BIGcovtype);
                elseif strcmp(options.covtype,'diag')
                    prec = gc * (ndim+0.1-1) * (1./range_data);
                    Smc = inv(prec + (1 ./ ((range_data/2).^2) ));
                    mc = Smc .* prec .* sum(subj_m_init(:,I(1:ii),k),2)' / gc ;  % what if zeromean????
                    cc = diag(sum(subj_gram_init(:,I(1:ii),k),2));
                    centroids_init(k) = centroid_new(cc,gc,mc,Smc,BIGcovtype);
                else % uniquefull or uniquediag
                    centroids_init(k) = centroid_new(gram_init,sumT,mc,Smc,BIGcovtype);
                end
            end
        end
        
        % set prior
        if cycle==1
            % prior for the covariance matrix
            prior = struct();
            prior.Omega = struct();
            if strcmp(options.covtype,'uniquediag') || strcmp(options.covtype,'diag')
                prior.Omega.Gam_shape = 0.5 * (ndim+0.1-1);
                prior.Omega.Gam_rate = 0.5 * range_data;
            else
                prior.Omega.Gam_shape = ndim+0.1-1;
                prior.Omega.Gam_rate = diag(range_data);
            end
            % prior for the mean
            prior.Mean = struct();
            prior.Mean.S = (range_data/2).^2;
            prior.Mean.iS = 1 ./ prior.Mean.S;
            prior.Mean.Mu = zeros(1,ndim);
        end
        
        % Having Gamma and the mean per centroid, we go on to get an
        % unbiased group estimation of the centroid covariance matrices
        for i = 1:N
            X = loadfile(files{i});
            load(strcat(BIGbase_tmpfolder,'/Gamma',num2str(i),'.mat')) % load Gamma
            for k=1:K
                E = X - repmat(centroids_init(k).m',size(X,1),1); % using the current mean estimation
                if strcmp(options.covtype,'full')
                    subj_gram_init(:,:,i,k) = ((E' .* repmat(Gam(:,k)',size(E,2),1)) * E);
                elseif strcmp(options.covtype,'diag')
                    subj_gram_init(:,i,k) = ( sum( (E.^2) .* repmat(Gam(:,k),1,size(E,2)) ) )';
                end
            end
        end
        for k=1:K
            gc = sum(subj_gamma_init(:,k));
            if strcmp(options.covtype,'full')
                centroids_init(k).rate = sum(subj_gram_init(:,:,:,k),3) + prior.Omega.Gam_rate;
                centroids_init(k).shape = gc + prior.Omega.Gam_shape;
            elseif strcmp(options.covtype,'diag')  % it is missing the estimation of the mean for uniqueXXX
                centroids_init(k).rate = sum(subj_gram_init(:,:,k),2)' + prior.Omega.Gam_rate;
                centroids_init(k).shape = gc + prior.Omega.Gam_shape;
            end
        end
            
        % Evaluate free energy (computing Gamma for the last time)
        for i = 1:N
            X = loadfile(files{i});
            % inference of state time courses
            data = struct('X',X,'C',NaN(size(X,1),K));
            hmm = loadhmm(hmm_i,T{i},K,centroids_init,P_init{i}, ...
                Pi_init{i},Dir2d_alpha_init{i},Dir_alpha_init{i},gram_init,prior);
            [Gam,~,Xi,l] = hsinference(data,T{i},hmm,[]); Gam = single(Gam);  
            %Gam = single([Gam zeros(size(Gam,1),K-size(Gam,2))]);
            save(strcat(BIGbase_tmpfolder,'/Gamma',num2str(i),'.mat'),'Gam')
            % transient probability
            hmm = hsupdate(Xi,Gam,T{i},hmm);
            P_init{i} = hmm.P; Pi_init{i} = hmm.Pi;
            Dir2d_alpha_init{i} = hmm.Dir2d_alpha; Dir_alpha_init{i} = hmm.Dir_alpha;
            loglik_init(i) = sum(l);
            subjfe_init(i,1) = - GammaEntropy(Gam,Xi,T{i},0);
            subjfe_init(i,2) = - GammaavLL(hmm,Gam,Xi,T{i});
            subjfe_init(i,3) = + KLtransition(hmm);
        end
        for k = 1:K
            statekl_init(k) = KLstate(centroids_init(k),prior,options.covtype,options.zeromean);
        end
        fe = - sum(loglik_init) + sum(subjfe_init(:)) + sum(statekl_init);
        
        if fe<best_fe
            best_fe = fe;
            subj_gamma = subj_gamma_init;
            subj_m = subj_m_init;
            subj_gram = subj_gram_init;
            gram = gram_init;
            centroids = centroids_init;
            hmm0 = hmm_i;
            subjfe(:,:,1) = subjfe_init(:,:,1);
            loglik(:,1) = loglik_init; 
            statekl(:,1) = statekl_init;
            P = P_init; Pi = Pi_init;
            Dir2d_alpha = Dir2d_alpha_init; Dir_alpha = Dir_alpha_init;
            fehist = best_fe;
        end
        
        fprintf('Init run %d, FE=%g (best=%g) \n',cycle,fe,best_fe);
        
    end
    
else % initial centroids specified by the user
    
    centroids = options.centroids; 
    subj_gamma = zeros(N,K);
    subj_m = zeros(ndim,N,K);
    if strcmp(options.covtype,'diag')
        subj_gram = zeros(ndim,N,K); gram = [];
    elseif strcmp(options.covtype,'full')
        subj_gram = zeros(ndim,ndim,N,K); gram = [];
    elseif strcmp(options.covtype,'uniquediag')
        gram = zeros(1,ndim); subj_gram = [];
    else % uniquefull
        gram = zeros(ndim,ndim); subj_gram = [];
    end
    
    P = cell(N,1); Pi = cell(N,1);
    Dir2d_alpha = cell(N,1); Dir_alpha = cell(N,1);

    % collect some stats
    for i = 1:N
        X = loadfile(files{i});
        if i==1
            options.inittype = 'random';
            options.cyc = 1;
            hmm0 = hmmmar(X,T{i},options);
            range_data = range(X);
        else
            range_data = max(range_data,range(X));
        end
        if strcmp(options.covtype,'uniquefull')
            gram = gram + X' * X;
        elseif strcmp(options.covtype,'uniquediag')
            gram = gram + sum(X.^2);
        end
    end
    
    % set prior
    % prior for the covariance matrix
    prior = struct();
    prior.Omega = struct();
    if strcmp(options.covtype,'uniquediag') || strcmp(options.covtype,'diag')
        prior.Omega.Gam_shape = 0.5 * (ndim+0.1-1);
        prior.Omega.Gam_rate = 0.5 * range_data;
    else
        prior.Omega.Gam_shape = ndim+0.1-1;
        prior.Omega.Gam_rate = diag(range_data);
    end
    % prior for the mean
    prior.Mean = struct();
    prior.Mean.S = (range_data/2).^2;
    prior.Mean.iS = 1 ./ prior.Mean.S;
    prior.Mean.Mu = zeros(1,ndim);
    
    % Init subject models and free energy computation
    for i = 1:N
        X = loadfile(files{i});
        data = struct('X',X,'C',NaN(size(X,1),K));
        hmm = loadhmm(hmm0,T{i},K,centroids,[],[],[],[],gram,prior);
        % get gamma
        [Gam,~,Xi,l] = hsinference(data,T{i},hmm,[]);
        % compute transition prob
        hmm = hsupdate(Xi,Gam,T{i},hmm);
        P{i} = hmm.P; Pi{i} = hmm.Pi;
        Dir2d_alpha{i} = hmm.Dir2d_alpha; Dir_alpha{i} = hmm.Dir_alpha;
        % compute free energy
        loglik(i,1) = sum(l);  
        subjfe(i,1,1) = - GammaEntropy(Gam,Xi,T{i},0); 
        subjfe(i,2,1) = - GammaavLL(hmm,Gam,Xi,T{i});
        subjfe(i,3,1) = + KLtransition(hmm);
        % subject meanv
        for k=1:K
            subj_gamma(i,k) = sum(Gam(:,k));
            E = X - repmat(centroids(k).m',size(X,1),1); % using the current mean estimation 
            if strcmp(options.covtype,'diag')
                subj_gram(:,i,k) = ( sum( (E.^2) .* repmat(Gam(:,k),1,size(E,2)) ) )';
            elseif strcmp(options.covtype,'full')
                subj_gram(:,:,i,k) =  ((E' .* repmat(Gam(:,k)',size(E,2),1)) * E);
            end
            if options.zeromean==0
                subj_m(:,i,k) = sum(repmat(Gam(:,k),1,size(X,2)) .* X)';  
            end
        end
    end
    
    for k = 1:K
        statekl(k,1) = KLstate(centroids(k),prior,options.covtype,options.zeromean);
    end
    fehist = sum(- loglik(:,1) + sum(sum(subjfe(:,:,1))) + sum(statekl(:,1)));
    
    fprintf('Cycle 1, free energy: %g \n',fehist);
  
end

clear centroid_gamma_init centroid_m_init centroid_gram_init centroids_init 
clear gram_init subjfe_init loglik_init statekl_init Pi_init P_init Dir2d_alpha_init Dir_alpha_init
    
% init stuff for stochastic learning
nUsed = zeros(1,N);
BIGbase_weights = BIGbase_weights * ones(1,N);
sampling_weights = BIGbase_weights;
if strcmp(options.covtype,'uniquefull') || strcmp(options.covtype,'uniquefull')
    ratec = gram + prior.Omega.Gam_rate;
    shapec = sumT + prior.Omega.Gam_shape;
end
undertol = 0;
Nrep = N/BIGNbatch;
% Stochastic learning
for cycle = 2:BIGcyc
      
    I = datasample(1:N,BIGNbatch,'Replace',false,'Weights',sampling_weights);
    nUsed(I) = nUsed(I) + 1;
    nUsed = nUsed - min(nUsed) + 1;
    sampling_weights = BIGbase_weights.^nUsed;
    
    % local parameters
    Gamma = cell(BIGNbatch,1); 
    for ii = 1:length(I)
        % load data
        i = I(ii);
        X = loadfile(files{i});        
        % inference of state time courses 
        data = struct('X',X,'C',NaN(size(X,1),K));
        hmm = loadhmm(hmm0,T{i},K,centroids,P{i},Pi{i},Dir2d_alpha{i},Dir_alpha{i},gram,prior);
        [Gamma{ii},~,Xi,l] = hsinference(data,T{i},hmm,[]); Gam = single(Gamma{ii});
        save(strcat(BIGbase_tmpfolder,'/Gamma',num2str(i),'.mat'),'Gam')
        %if (-loglik(i,cycle-1)+sum(subjfe(i,1:2,cycle-1))) < (-loglik(i,cycle)+sum(subjfe(i,1:2,cycle)))
        %    fprintf('1\n'); pause(0.5)
            %if cycle>=3, keyboard; end
        %end;  
        % transition prob estimation
        hmm = hsupdate(Xi,Gamma{ii},T{i},hmm);
        P{i} = hmm.P; Pi{i} = hmm.Pi; % one per subject, not like group HMM
        Dir2d_alpha{i} = hmm.Dir2d_alpha; Dir_alpha{i} = hmm.Dir_alpha;
        % elements of free energy
        loglik(i,cycle) = sum(l);
        subjfe(i,1,cycle) = - GammaEntropy(Gamma{ii},Xi,T{i},0); 
        subjfe(i,2,cycle) = - GammaavLL(hmm,Gamma{ii},Xi,T{i}); 
        subjfe(i,3,cycle) = + KLtransition(hmm);
        %if (-loglik(i,cycle-1)+sum(subjfe(i,1:3,cycle-1))) < (-loglik(i,cycle)+sum(subjfe(i,1:3,cycle)))
        %    fprintf('2\n'); pause(0.5)
        %end
        % compute the statistics of the mean, to avoid another loop over subjects
        for k=1:K
            subj_gamma(i,k) = sum(Gamma{ii}(:,k));
            if options.zeromean==0
                subj_m(:,i,k) = sum(repmat(Gamma{ii}(:,k),1,size(X,2)) .* X)';  
            end
        end
    end
    for i = setdiff(1:N,I)
        loglik(i,cycle) = loglik(i,cycle-1);
        for j = 1:3
            subjfe(i,j,cycle) = subjfe(i,j,cycle-1);
        end
    end
        
    % update covariance matrix statistic
    for ii = 1:length(I)
        % load data
        i = I(ii);
        X = loadfile(files{i}); 
        for k=1:K
            E = X - repmat(centroids(k).m',size(X,1),1); % using the current mean estimation 
            if strcmp(options.covtype,'full')
                subj_gram(:,:,i,k) = ((E' .* repmat(Gamma{ii}(:,k)',size(E,2),1)) * E);
            elseif strcmp(options.covtype,'diag')
                subj_gram(:,i,k) = ( sum( (E.^2) .* repmat(Gamma{ii}(:,k),1,size(E,2)) ) )';
            end
        end
    end
    
    % Update the global centroids, and collect centroid free energy
    rho = (cycle + BIGdelay)^(-BIGforgetrate);
    for k=1:K
        gc = Nrep * sum(subj_gamma(I,k));
        % so, here, both mean and covariance matrix are updated using
        % the info of the last iteration's centroid
        if strcmp(options.covtype,'full')
            ratec = Nrep * sum(subj_gram(:,:,I,k),3) + prior.Omega.Gam_rate;
            iratec = inv(ratec);
            shapec = gc + prior.Omega.Gam_shape;
            prec = shapec * iratec * gc;
            Smc = inv(prec + diag(prior.Mean.iS));
            mc = Smc * prec * Nrep * sum(subj_m(:,I,k),2) / gc;
        elseif strcmp(options.covtype,'diag')  % it is missing the estimation of the mean for uniqueXXX
            ratec = Nrep * sum(subj_gram(:,I,k),2)' + prior.Omega.Gam_rate;
            iratec = 1 ./ ratec; 
            shapec = gc + prior.Omega.Gam_shape;
            prec = shapec * iratec * gc;
            Smc = inv(prec + prior.Mean.iS);
            mc = Smc .* prec .* Nrep * sum(subj_m(:,I,k),2)' / gc ;
        end
        centroids(k) = centroid_update(centroids(k),ratec,shapec,mc,Smc,rho);
        statekl(k,cycle) = KLstate(centroids(k),prior,options.covtype,options.zeromean);
    end
    
    %loglik_temp = XavLL(files,T,centroids,BIGGamma);
    %if (-sum(loglik(:,cycle))+sum(statekl(:,cycle-1))) < (-sum(loglik_temp)+sum(statekl(:,cycle)))
    %        fprintf('3\n'); pause(0.5)
    %end;
    
    fehist = [ fehist; (-sum(loglik(:,cycle)) + sum(statekl(:,cycle)) + sum(sum(subjfe(:,:,cycle)))) ];
        
    ch = (fehist(end)-fehist(end-1)) / abs(fehist(end)-fehist(1));
    fprintf('Cycle %d, free energy: %g (relative change %g) \n',cycle,fehist(end),ch);
    if cycle>5 && abs(ch) < BIGtol && cycle>BIGmincyc, 
        undertol = undertol + 1; 
    else
        undertol = 0;
    end
    if undertol > BIGundertol_tostop, break; end
        
end

subj_stats = struct('subj_gamma',subj_gamma,'subj_m',subj_m,'subj_gram',subj_gram);
markovTrans = struct();
markovTrans.P = P;
markovTrans.Pi = Pi;
markovTrans.Dir2d_alpha = Dir2d_alpha;
markovTrans.Dir_alpha = Dir_alpha;

end


function centroid = centroid_new(rate,shape,m,Sm,type)
centroid.m = m(:);
centroid.Sm = Sm;
centroid.rate = rate;
centroid.shape = shape;
if isvector(centroid.rate)
    centroid.irate = 1 ./ centroid.rate;
else
    centroid.irate = inv(centroid.rate);
end
centroid.type = type;
end


function centroid = centroid_update(centroid,rate,shape,m,Sm,rho)
centroid.m = (1-rho) * centroid.m + rho * m(:);
centroid.Sm = (1-rho) * centroid.Sm + rho * Sm;
centroid.rate = (1-rho) * centroid.rate + rho * rate;
centroid.shape = (1-rho) * centroid.shape + rho * shape;
if isvector(centroid.rate)
    centroid.irate = 1 ./ centroid.rate;
else
    centroid.irate = inv(centroid.rate);
end
end


function kl = symm_kl_div(c1,c2)
% symmetric kullback leibler divergence between MVN c1 and c2
N = length(c1.m);
m = c1.m - c2.m; 
mm = m*m';
cov1 = c1.rate / c1.shape; icov1 = c1.irate * c1.shape;
cov2 = c2.rate / c2.shape; icov2 = c2.irate * c2.shape;
if isvector(cov1)
    cov1 = diag(cov1); icov1 = diag(icov1); 
    cov2 = diag(cov2); icov2 = diag(icov2); 
end
sicov = icov1 + icov2;
kl = (cov1(:).'*icov2(:) + cov2(:).'*icov1(:) + mm(:).'*sicov(:) - 2*N) / 4; 
% this can be done more efficiently
end


function Entr = GammaEntropy(Gamma,Xi,T,order)
% Entropy of the state time courses
Entr = 0; K = size(Gamma,2);
for tr=1:length(T);
    t = sum(T(1:tr-1)) - (tr-1)*order + 1;
    Gamma_nz = Gamma(t,:); Gamma_nz(Gamma_nz==0) = realmin;
    Entr = Entr - sum(Gamma_nz.*log(Gamma_nz));
    t = (sum(T(1:tr-1)) - (tr-1)*(order+1) + 1) : ((sum(T(1:tr)) - tr*(order+1)));
    Xi_nz = Xi(t,:,:); Xi_nz(Xi_nz==0) = realmin;
    Psi=zeros(size(Xi_nz));                    % P(S_t|S_t-1)
    for k = 1:K,
        sXi = sum(permute(Xi_nz(:,k,:),[1 3 2]),2);
        Psi(:,k,:) = Xi_nz(:,k,:)./repmat(sXi,[1 1 K]); 
    end;
    Psi(Psi==0) = realmin;
    Entr = Entr - sum(Xi_nz(:).*log(Psi(:)));    % entropy of hidden states
end

end


function avLL = GammaavLL(hmm,Gamma,Xi,T)
% average loglikelihood for state time course
avLL = 0; K = length(hmm.state);
jj = zeros(length(T),1); % reference to first time point of the segments
for in=1:length(T);
    jj(in) = sum(T(1:in-1)) + 1;
end
PsiDir_alphasum = psi(sum(hmm.Dir_alpha,2));
for l=1:K,
    % avLL initial state  
    avLL = avLL + sum(Gamma(jj,l)) * (psi(hmm.Dir_alpha(l)) - PsiDir_alphasum);
end     
% avLL remaining time points  
PsiDir2d_alphasum=psi(sum(hmm.Dir2d_alpha(:)));
for k=1:K,
    for l=1:K,
        avLL = avLL + sum(Xi(:,l,k)) * (psi(hmm.Dir2d_alpha(l,k))-PsiDir2d_alphasum);
    end
end
end


function avLL = XavLL(files,T,centroids,Gamma)
X = loadfile(files{1}); N = length(files);
avLL = zeros(N,1); ndim = size(X,2); K = length(centroids);
ltpi = ndim/2 * log(2*pi);
for i = 1:N
    X = loadfile(files{i});
    for k=1:K
        centroid = centroids(k);
        if isvector(centroid.rate)
            ldetWishB=0;
            PsiWish_alphasum=0;
            for n=1:ndim,
                ldetWishB=ldetWishB+0.5*log(centroid.rate(n));
                PsiWish_alphasum=PsiWish_alphasum+0.5*psi(centroid.shape);
            end;
            C = centroid.shape ./ centroid.rate;
            avLL(i) = avLL(i) + sum(Gamma{i}(:,k)) * (-ltpi-ldetWishB+PsiWish_alphasum);
            NormWishtrace = 0.5 * sum(centroid.Sm' .* C);
        else 
            ldetWishB=0.5*logdet(centroid.rate);
            PsiWish_alphasum=0;
            for n=1:ndim
                PsiWish_alphasum=PsiWish_alphasum+0.5*psi(centroid.shape/2+0.5-n/2);
            end;
            C = centroid.shape * centroid.irate;
            avLL(i) = avLL(i) + sum(Gamma{i}(:,k)) * (-ltpi-ldetWishB+PsiWish_alphasum);
            NormWishtrace = 0.5 * sum(sum(centroid.Sm .* C));
        end
        meand = repmat(centroid.m',size(X,1),1);
        d = X - meand;
        if isvector(centroid.rate)
            Cd =  repmat(C',1,sum(T{i})) .* d';
        else
            Cd = C * d';
        end
        dist=zeros(sum(T{i}),1);
        for n=1:ndim,
            dist=dist-0.5*d(:,n).*Cd(n,:)';
        end
        avLL(i) = avLL(i) + sum(Gamma{i}(:,k).*(dist - NormWishtrace));
    end
end
end

function KLdiv = KLtransition(hmm)
% KL divergence for the transition and initial probabilities
KLdiv = dirichlet_kl(hmm.Dir_alpha,hmm.prior.Dir_alpha); 
K = length(hmm.state);
for l=1:K,
    % KL-divergence for transition prob
    KLdiv = KLdiv + dirichlet_kl(hmm.Dir2d_alpha(l,:),hmm.prior.Dir2d_alpha(l,:));
end
end


function KLdiv = KLstate(centroid,prior,covtype,zeromean) 
% KL divergence between a state and its prior (cov and mean)
KLdiv = 0; 
% cov matrix
if strcmp(covtype,'full') 
    KLdiv = wishart_kl(centroid.rate,prior.Omega.Gam_rate, ...
                centroid.shape,prior.Omega.Gam_shape);
elseif strcmp(covtype,'diag')
    ndim = length(centroid.rate);
    for n=1:ndim
        KLdiv = KLdiv + gamma_kl(centroid.shape,prior.Omega.Gam_shape, ...
                    centroid.rate(n),prior.Omega.Gam_rate(n));
    end
end 
% mean
if ~zeromean 
    ndim = length(centroid.m);
    if strcmp(covtype,'full') 
        KLdiv = KLdiv + gauss_kl(centroid.m,zeros(ndim,1),centroid.Sm,diag(prior.Mean.S));
    else
        for n=1:ndim
            KLdiv = KLdiv + gauss_kl(centroid.m(n),0,centroid.Sm(n),prior.Mean.S(n));
        end
    end
end
end

function state = fit_state(files,TT,BIGGamma,hmm)
N = length(files); K = length(hmm.state);
T = []; X = [];
Gamma = [];
for i = 1:N
    X = [X; loadfile(files{i})];
    T = [T TT{i}];
    Gamma = [Gamma; BIGGamma{i}]; 
end
XXGXX = cell(K,1);
setxx;
Tres = sum(T);
Gammasum = sum(Gamma);
[hmm,XW] = updateW(hmm,Gamma,X,XX,XXGXX);
hmm = updateOmega(hmm,Gamma,Gammasum,X,Tres,XX,XXGXX,XW);
state = hmm.state;
end