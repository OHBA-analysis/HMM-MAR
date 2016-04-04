function [metahmm,markovTrans,fehist,feterms,rho] = trainBigHMM(Xin,T,options)
% 1) Initialize BigHMM by estimating a separate HMM on each file,
%    and then using an adapted version of kmeans to clusterize the states
%    (see initBigHMM)
% 2) Run the BigHMM algorithm
%
% INPUTS
% Xin: cell with strings referring to the files containing each subject's data, 
%       or cell with with matrices (time points x channels) with each
%       subject's data
% T: cell of vectors, where each element has the length of each trial per
%       subject. Dimension of T{n} has to be (1 x nTrials)
% options: HMM options for both the subject and the group runs
%
% Diego Vidaurre, OHBA, University of Oxford (2016)

N = length(Xin);

% Check parameters
checkBigHMMparameters;
X = loadfile(Xin{1},T{1}); ndim = size(X,2); XW = [];
subjfe = zeros(N,3); % Gamma entropy, Gamma loglik and KL-divergences of transition prob. mat.
loglik = zeros(N,1); % data log likelihood
statekl = 0; % state KL-divergences
Dir2d_alpha = zeros(K,K,N); Dir_alpha = zeros(K,N);
fehist = 0; rho = 0;

% Initialization
initBigHMM;

% init stuff for stochastic learning
nUsed = zeros(1,N); 
BIGbase_weights = BIGbase_weights * ones(1,N);
sampling_weights = BIGbase_weights;
undertol = 0;
orders = formorders(options.order,options.orderoffset,options.timelag,options.exptimelag);

% load('/tmp/debugBigHMM.mat'); gramm = [];
% fprintf('Init run %d, FE=%g (best=%g) \n',cycle,fe,best_fe);
% % load('/tmp/options_bighmm'); BIGNbatch = options_bighmm.BIGNbatch; 
% BIGcyc = 25; BIGNbatch = N ; % N = 1; 

Tfactor = N/BIGNbatch; 

%[ sum(loglik(:,1)) squeeze(sum(subjfe(:,1,1),1)) squeeze(sum(subjfe(:,2,1),1)) squeeze(sum(subjfe(:,3,1),1))]

% Stochastic learning
for cycle = 2:BIGcyc
    
    % sampling batch
    I = datasample(1:N,BIGNbatch,'Replace',false,'Weights',sampling_weights);
    %I = 1:BIGNbatch; %I = 1:N;
    nUsed(I) = nUsed(I) + 1;
    nUsed = nUsed - min(nUsed) + 1;
    sampling_weights = BIGbase_weights.^nUsed;
        
    % read data for this batch
    Tbatch = [];
    for ii = 1:length(I), i = I(ii); Tbatch = [Tbatch T{i}]; end
    X = zeros(sum(Tbatch),ndim);
    XX = cell(1); 
    XX{1} = zeros(sum(Tbatch)-length(Tbatch)*options.order,length(orders)*ndim+(~options.zeromean));
    Y = zeros(sum(Tbatch)-length(Tbatch)*options.order,ndim);
    tacc = 0; t2acc = 0;
    for ii = 1:length(I)
        i = I(ii);
        t = (1:sum(T{i})) + tacc; t2 = (1:(sum(T{i})-length(T{i})*options.order)) + t2acc;
        [X(t,:),XX{1}(t2,:),Y(t2,:)]  = loadfile(Xin{i},T{i},options);
        tacc = tacc + sum(T{i}); t2acc = t2acc + sum(T{i}) - length(T{i})*options.order;
    end
    
    % local parameters (Gamma, Xi, P, Pi, Dir2d_alpha and Dir_alpha)
    tacc = 0; t2acc = 0;
    Gamma = cell(BIGNbatch,1); Xi = cell(BIGNbatch,1);
    XXGXX = cell(K,1);  
    for k=1:K, XXGXX{k} = zeros(size(XX{1},2)); end
    for ii = 1:length(I)
        i = I(ii); 
        t = (1:sum(T{i})) + tacc; t2 = (1:(sum(T{i})-length(T{i})*options.order)) + t2acc;
        tacc = tacc + sum(T{i}); t2acc = t2acc + sum(T{i}) - length(T{i})*options.order;
        data = struct('X',X(t,:),'C',NaN(sum(T{i})-length(T{i})*options.order,K));
        XX_i = cell(1); XX_i{1} = XX{1}(t2,:); Y_i = Y(t2,:);
        if BIGuniqueTrans
            metahmm_i = metahmm;
        else
            metahmm_i = copyhmm(metahmm,P(:,:,i),Pi(:,i)',Dir2d_alpha(:,:,i),Dir_alpha(:,i)');
        end
        [Gamma{ii},~,Xi{ii}] = hsinference(data,T{i},metahmm_i,Y_i,[],XX_i);
        for k=1:K
            XXGXX{k} = XXGXX{k} + (XX_i{1}' .* repmat(Gamma{ii}(:,k)',size(XX_i{1},2),1)) * XX_i{1};
        end
        metahmm_i = hsupdate(Xi{ii},Gamma{ii},T{i},metahmm_i);
        if BIGuniqueTrans
            Dir_alpha(:,i) = 0;
            for trial=1:length(T{i})
                t3 = sum(T{i}(1:trial-1)) - options.order*(trial-1) + 1;
                Dir_alpha(:,i) = Dir_alpha(:,i) + Gamma{ii}(t3,:)';
            end
            Dir2d_alpha(:,:,i) = squeeze(sum(Xi{ii},1));
            subjfe(i,1:2,cycle) = evalfreeenergy([],T{i},Gamma{ii},Xi{ii},metahmm_i,[],[],[1 0 1 0 0]); % Gamma entropy&LL
        else
            P(:,:,i) = metahmm_i.P; Pi(:,i) = metahmm_i.Pi'; % one per subject, not like pure group HMM
            Dir2d_alpha(:,:,i) = metahmm_i.Dir2d_alpha; Dir_alpha(:,i) = metahmm_i.Dir_alpha';
            subjfe(i,:,cycle) = evalfreeenergy([],T{i},Gamma{ii},Xi{ii},metahmm_i,[],[],[1 0 1 1 0]); % + transitions LL
        end
    end
    if BIGuniqueTrans
        metahmm.Dir_alpha = sum(Dir_alpha,2)' + Dir_alpha_prior;
        metahmm.Dir2d_alpha = sum(Dir2d_alpha,3) + Dir2d_alpha_prior;
        [metahmm.P,metahmm.Pi] = computePandPi(metahmm.Dir_alpha,metahmm.Dir2d_alpha);
        subjfe(:,3,cycle) = evalfreeenergy([],[],[],[],metahmm,[],[],[0 0 0 1 0]) / N; % "shared" KL
    end
        
    % global parameters (metahmm), and collect metastate free energy
    rho(cycle) = (cycle + BIGdelay)^(-BIGforgetrate); 
    MGamma = cell2mat(Gamma);
    % W
    if isfield(metahmm.state(1),'W') && ~isempty(metahmm.state(1).W.Mu_W)
        [metahmm_noisy,XW] = updateW(metahmm,MGamma,Y,XX,XXGXX,Tfactor);
        metahmm = metastates_update(metahmm,metahmm_noisy,rho(cycle),1);
    end
    % Omega 
    if isfield(metahmm.state(1),'Omega') || isfield(metahmm,'Omega')  
        metahmm_noisy = updateOmega(metahmm,MGamma,sum(MGamma),Y,Tbatch,XX,XXGXX,XW,Tfactor);
        metahmm = metastates_update(metahmm,metahmm_noisy,rho(cycle),2);
    end    
    % sigma
    if ~isempty(orders)
        metahmm_noisy = updateSigma(metahmm);
        metahmm = metastates_update(metahmm,metahmm_noisy,rho(cycle),3);
    end
    % alpha
    if ~isempty(orders)
        metahmm_noisy = updateAlpha(metahmm);
        metahmm = metastates_update(metahmm,metahmm_noisy,rho(cycle),4);
    end       
   
    % rest of the free energy (states' KL and data loglikelihood)
    [fe,ll] = evalfreeenergy(X,Tbatch,MGamma,cell2mat(Xi),metahmm,Y,XX,[0 1 0 0 1]); % state KL
    statekl(1,cycle) = sum(fe(2:end));
    tacc = 0;
    for ii = 1:length(I)
        i = I(ii); 
        t = (1:sum(T{i})-length(T{i})*options.order) + tacc; tacc = tacc + length(t);
        loglik(i,cycle) = sum(ll(t));
    end
    
    % bring from last iteration whatever was not updated
    for i = setdiff(1:N,I)
        loglik(i,cycle) = loglik(i,cycle-1);
        if ~BIGuniqueTrans
            subjfe(i,:,cycle) = subjfe(i,:,cycle-1);
        end
    end
        
    %[ sum(loglik(:,cycle)) squeeze(sum(subjfe(:,1,cycle),1)) squeeze(sum(subjfe(:,2,cycle),1)) squeeze(sum(subjfe(:,3,cycle),1))]

    fehist(cycle) = (-sum(loglik(:,cycle)) + statekl(1,cycle) + sum(sum(subjfe(:,:,cycle))));
    ch = (fehist(end)-fehist(end-1)) / abs(fehist(end)-fehist(1));
    if BIGverbose
        fprintf('Cycle %d, free energy: %g (relative change %g), rho: %g \n',cycle,fehist(end),ch,rho(cycle));
    end

    if cycle>5 && abs(ch) < BIGtol && cycle>BIGmincyc, 
        undertol = undertol + 1; 
    else
        undertol = 0;
    end
    if undertol > BIGundertol_tostop, break; end
    
end

markovTrans = struct();
markovTrans.P = P;
markovTrans.Pi = Pi;
markovTrans.Dir2d_alpha = Dir2d_alpha;
markovTrans.Dir_alpha = Dir_alpha;
markovTrans.prior.Dir2d_alpha = metahmm.prior.Dir2d_alpha;
markovTrans.prior.Dir_alpha = metahmm.prior.Dir_alpha;

if ~BIGuniqueTrans
    metahmm.Dir_alpha = sum(Dir_alpha,2)' + Dir_alpha_prior;
    metahmm.Dir2d_alpha = sum(Dir2d_alpha,3) + Dir2d_alpha_prior;
    [metahmm.P,metahmm.Pi] = computePandPi(metahmm.Dir_alpha,metahmm.Dir2d_alpha);
end

feterms = struct();
feterms.loglik = loglik;
feterms.subjfe = subjfe;
feterms.statekl = statekl;

end


