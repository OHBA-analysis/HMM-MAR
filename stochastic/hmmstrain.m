function [metahmm,markovTrans,fehist,feterms,rho] = hmmstrain(Xin,T,metahmm,info,options)
% Stochastic HMM variational inference
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

N = length(Xin); K = length(metahmm.state);
X = loadfile(Xin{1},T{1},options); ndim = size(X,2); XW = [];
subjfe = info.subjfe;
loglik = info.loglik;
statekl = info.statekl;
Dir2d_alpha = info.Dir2d_alpha;
Dir_alpha = info.Dir_alpha;
fehist = info.fehist; 
metahmm_best = metahmm;
Dir2d_alpha_best = Dir2d_alpha; 
Dir_alpha_best = Dir_alpha;
cyc_best = 1;
tp_less = max(options.embeddedlags) + max(-options.embeddedlags);

clear info;

% init stochastic learning stuff
nUsed = zeros(1,N); 
sampling_weights = options.BIGbase_weights;
undertol = 0; count = 0;
orders = formorders(options.order,options.orderoffset,options.timelag,options.exptimelag);
Tfactor = N/options.BIGNbatch; 

% Stochastic learning
for cycle = 2:options.BIGcyc
    
    % sampling batch
    I = datasample(1:N,options.BIGNbatch,'Replace',false,'Weights',sampling_weights);
    nUsed(I) = nUsed(I) + 1;
    nUsed = nUsed - min(nUsed) + 1;
    sampling_weights = options.BIGbase_weights.^nUsed;
        
    % read data for this batch
    Tbatch = []; Tbatch_list = cell(length(I),1);
    for ii = 1:length(I), 
        i = I(ii); 
        Tbatch = [Tbatch (T{i}-tp_less)]; 
        Tbatch_list{ii} = (T{i}-tp_less);
    end
    X = zeros(sum(Tbatch),ndim);
    XX = cell(1); 
    XX{1} = zeros(sum(Tbatch)-length(Tbatch)*options.order,length(orders)*ndim+(~options.zeromean));
    Y = zeros(sum(Tbatch)-length(Tbatch)*options.order,ndim);
    tacc = 0; t2acc = 0;
    for ii = 1:length(I)
        i = I(ii);
        [X_ii,XX_ii,Y_ii,T_ii]  = loadfile(Xin{i},T{i},options);
        t = (1:sum(T_ii)) + tacc;
        t2 = (1:(sum(T_ii)-length(T_ii)*options.order)) + t2acc;
        X(t,:) = X_ii; XX{1}(t2,:) = XX_ii; Y(t2,:) = Y_ii;
        tacc = tacc + sum(T_ii); t2acc = t2acc + sum(T_ii) - length(T_ii)*options.order;
    end
    
    % local parameters (Gamma, Xi, P, Pi, Dir2d_alpha and Dir_alpha),
    % and free energy relative these local parameters
    tacc = 0; t2acc = 0;
    Gamma = cell(options.BIGNbatch,1); Xi = cell(options.BIGNbatch,1);
    XXGXX = cell(K,1);  
    for k=1:K, XXGXX{k} = zeros(size(XX{1},2)); end
    for ii = 1:length(I)
        i = I(ii); 
        t = (1:sum(Tbatch_list{ii})) + tacc; 
        t2 = (1:(sum(Tbatch_list{ii})-length(Tbatch_list{ii})*options.order)) + t2acc;
        tacc = tacc + sum(Tbatch_list{ii}); 
        t2acc = t2acc + sum(Tbatch_list{ii}) - length(Tbatch_list{ii})*options.order;
        data = struct('X',X(t,:),'C',NaN(sum(Tbatch_list{ii})-length(Tbatch_list{ii})*options.order,K));
        XX_i = cell(1); XX_i{1} = XX{1}(t2,:); Y_i = Y(t2,:);
        if options.BIGuniqueTrans
            metahmm_i = metahmm;
        else
            metahmm_i = copyhmm(metahmm,P(:,:,i),Pi(:,i)',Dir2d_alpha(:,:,i),Dir_alpha(:,i)');
        end
        [Gamma{ii},~,Xi{ii}] = hsinference(data,Tbatch_list{ii},metahmm_i,Y_i,[],XX_i); % state time courses
        for k=1:K
            XXGXX{k} = XXGXX{k} + (XX_i{1}' .* repmat(Gamma{ii}(:,k)',size(XX_i{1},2),1)) * XX_i{1};
        end
        if options.BIGuniqueTrans % update transition probabilities
            Dir_alpha(:,i) = 0;
            for trial=1:length(Tbatch_list{ii})
                t3 = sum(Tbatch_list{ii}(1:trial-1)) - options.order*(trial-1) + 1;
                Dir_alpha(:,i) = Dir_alpha(:,i) + Gamma{ii}(t3,:)';
            end
            Dir2d_alpha(:,:,i) = squeeze(sum(Xi{ii},1));
        else
            metahmm_i = hsupdate(Xi{ii},Gamma{ii},Tbatch_list{ii},metahmm_i);
            P(:,:,i) = metahmm_i.P; Pi(:,i) = metahmm_i.Pi'; % one per subject, not like pure group HMM
            Dir2d_alpha(:,:,i) = metahmm_i.Dir2d_alpha; Dir_alpha(:,i) = metahmm_i.Dir_alpha';
            subjfe(i,:,cycle) = evalfreeenergy([],Tbatch_list{ii},Gamma{ii},Xi{ii},metahmm_i,[],[],[1 0 1 1 0]); 
        end
    end
    if options.BIGuniqueTrans
        metahmm.Dir_alpha = sum(Dir_alpha,2)' + metahmm.prior.Dir_alpha_prior;
        metahmm.Dir2d_alpha = sum(Dir2d_alpha,3) + metahmm.prior.Dir2d_alpha_prior;
        [metahmm.P,metahmm.Pi] = computePandPi(metahmm.Dir_alpha,metahmm.Dir2d_alpha);
        subjfe(:,3,cycle) = evalfreeenergy([],[],[],[],metahmm,[],[],[0 0 0 1 0]) / N; % "shared" P/Pi KL
        for ii = 1:length(I)
            i = I(ii); % Gamma entropy&LL
            subjfe(i,1:2,cycle) = evalfreeenergy([],Tbatch_list{ii},Gamma{ii},Xi{ii},metahmm,[],[],[1 0 1 0 0]); 
        end
    end
        
    % global parameters (metahmm), and collect metastate free energy
    rho(cycle) = (cycle + options.BIGdelay)^(-options.BIGforgetrate); 
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
        t = (1:sum(Tbatch_list{ii})-length(Tbatch_list{ii})*options.order) + tacc; tacc = tacc + length(t);
        loglik(i,cycle) = sum(ll(t));
    end
    
    % bring from last iteration whatever was not updated
    for i = setdiff(1:N,I)
        loglik(i,cycle) = loglik(i,cycle-1);
        if ~options.BIGuniqueTrans
            subjfe(i,:,cycle) = subjfe(i,:,cycle-1);
        end
    end
          
    fehist(cycle) = (-sum(loglik(:,cycle)) + statekl(1,cycle) + sum(sum(subjfe(:,:,cycle))));
    ch = (fehist(end)-fehist(end-1)) / abs(fehist(end)-fehist(1));
    if min(fehist)==fehist(cycle)
        metahmm_best = metahmm; 
        cyc_best = cycle;
        if ~options.BIGuniqueTrans, P_best = P; Pi_best = Pi; end
        Dir2d_alpha_best = Dir2d_alpha; Dir_alpha_best = Dir_alpha;
        count = 0;
    else
        count = count + 1; 
    end
    if options.BIGverbose
        fprintf('Cycle %d, free energy: %g (relative change %g), rho: %g \n', ...
            cycle,fehist(end),ch,rho(cycle));
    end

    if cycle>5 && abs(ch) < options.BIGtol 
        undertol = undertol + 1; 
    else
        undertol = 0;
    end
    if cycle>options.BIGmincyc
        if undertol > options.BIGundertol_tostop, break; end
        if count >= options.BIGcycnobetter_tostop, break; end
    end
    
end

metahmm = metahmm_best;
markovTrans = struct();
if ~options.BIGuniqueTrans
    markovTrans.P = P_best;
    markovTrans.Pi = Pi_best;
end
markovTrans.Dir2d_alpha = Dir2d_alpha_best;
markovTrans.Dir_alpha = Dir_alpha_best;
markovTrans.prior.Dir2d_alpha = metahmm.prior.Dir2d_alpha;
markovTrans.prior.Dir_alpha = metahmm.prior.Dir_alpha;
fehist = fehist(1:cyc_best);
loglik = loglik(:,1:cyc_best);
subjfe = subjfe(:,:,1:cyc_best);
statekl = statekl(1:cyc_best);
rho = rho(1:cyc_best);

if ~options.BIGuniqueTrans
    metahmm.Dir_alpha = sum(markovTrans.Dir_alpha,2)' + metahmm.prior.Dir_alpha_prior;
    metahmm.Dir2d_alpha = sum(markovTrans.Dir2d_alpha,3) + metahmm.prior.Dir2d_alpha_prior;
    [metahmm.P,metahmm.Pi] = computePandPi(metahmm.Dir_alpha,metahmm.Dir2d_alpha);
end

feterms = struct();
feterms.loglik = loglik;
feterms.subjfe = subjfe;
feterms.statekl = statekl;
 
if metahmm.train.BIGverbose
    fprintf('Model: %d states, %d subjects, batch size %d, covariance: %s \n', ...
        K,length(T),options.BIGNbatch,metahmm.train.covtype);
    if metahmm.train.exptimelag>1,
        fprintf('Exponential lapse: %g, order %g, offset %g \n', ...
            metahmm.train.exptimelag,metahmm.train.order,metahmm.train.orderoffset)
    else
        fprintf('Lapse: %d, order %g, offset %g \n', ...
            metahmm.train.timelag,metahmm.train.order,metahmm.train.orderoffset)
    end
end

end


