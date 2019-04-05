function [hmm,info] = hmmsinith(Xin,T,options,hmm)
% Initialisation before stochastic HMM variational inference, when an initial 
% hmm structure is provided
%
% INPUTS
% Xin: cell with strings referring to the files containing each subject's data,
%       or cell with with matrices (time points x channels) with each
%       subject's data
% T: cell of vectors, where each element has the length of each trial per
%       subject. Dimension of T{n} has to be (1 x nTrials)
% options: HMM options for both the subject and the group runs
% hmm: the initial HMM structure
%
% Diego Vidaurre, OHBA, University of Oxford (2016)

N = length(T); K = length(hmm.state);
subjfe_init = zeros(N,3);
loglik_init = zeros(N,1);
info = struct();

Dir2d_alpha_init = zeros(K,K,N); Dir_alpha_init = zeros(K,N);
for i = 1:N
    % read data
    [X,XX,Y,Ti] = loadfile(Xin{i},T{i},options);
    if i==1 % complete a few things
        if isfield(options,'hmm')
            hmm.train = rmfield(options,'hmm');
        else
            hmm.train = options; 
        end
        hmm.train.ndim = size(Y,2);
        hmm.train.active = ones(1,K);
        hmm.train.orders = formorders(hmm.train.order,hmm.train.orderoffset,...
            hmm.train.timelag,hmm.train.exptimelag);
        if hmm.train.pcapred>0
            hmm.train.Sind = true(hmm.train.pcapred,size(Y,2));
        else
            hmm.train.Sind = formindexes(hmm.train.orders,hmm.train.S)==1;
        end
        if ~hmm.train.zeromean, hmm.train.Sind = [true(1,hmm.train.ndim); hmm.train.Sind]; end
        Dir2d_alpha = hmm.Dir2d_alpha; Dir_alpha = hmm.Dir_alpha; P = hmm.P; Pi = hmm.Pi;
        if isfield(hmm,'prior'), hmm = rmfield(hmm,'prior'); end
        hmm = hmmhsinit(hmm); % set priors
        hmm.Dir2d_alpha = Dir2d_alpha; hmm.Dir_alpha = Dir_alpha; hmm.P = P; hmm.Pi = Pi;
    end

    [Gamma,~,Xi] = hsinference(X,Ti,hmm,Y,options,XX);
    checkGamma(Gamma,Ti,hmm.train,i);
    for trial=1:length(Ti)
        t = sum(Ti(1:trial-1)) + 1;
        Dir_alpha_init(:,i) = Dir_alpha_init(:,i) + Gamma(t,:)';
    end
    Dir2d_alpha_init(:,:,i) = squeeze(sum(Xi,1));
    loglik_init(i) = -evalfreeenergy([],Ti,Gamma,[],hmm,Y,XX,[0 1 0 0 0]); % data LL
    subjfe_init(i,1:2) = evalfreeenergy([],Ti,Gamma,Xi,hmm,[],[],[1 0 1 0 0]); % Gamma entropy&LL
end

subjfe_init(:,3) = evalfreeenergy([],[],[],[],hmm,[],[],[0 0 0 1 0]) / N; % "share" P and Pi KL
statekl_init = sum(evalfreeenergy([],[],[],[],hmm,[],[],[0 0 0 0 1])); % state KL
fe = - sum(loglik_init) + sum(subjfe_init(:)) + statekl_init;

info.Dir2d_alpha = Dir2d_alpha_init; 
info.Dir_alpha = Dir_alpha_init;
info.subjfe = subjfe_init;
info.loglik = loglik_init;
info.statekl = statekl_init;
info.fehist = (-sum(info.loglik) + sum(info.statekl) + sum(sum(info.subjfe)));

if options.BIGverbose
    fprintf('Init, free energy = %g \n',fe);
end

end
