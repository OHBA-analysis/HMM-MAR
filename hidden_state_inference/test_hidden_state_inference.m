%TEST_HIDDEN_STATE_INFERENCE
%
% Use to check functionality of mex interface


% set up generator and make data
rng(0, 'twister');

nSamples = 10000;
nClasses = 3;

Pi_0 = [0.2, 0.4, 0.4];
P    = [0.8, 0.1, 0.1; ...
	    0.3, 0.5, 0.2; ...
		0.1, 0.4, 0.5];

X = randn(nSamples, nClasses);
T = nSamples;

% set up HMM code
options = struct();
options.inittype = 'EM';
options.initcyc = 50;
options.initrep = 1;
options.K = nClasses;
options.order = 0; % changing this breaks Diego's code in ways I don't understand. 
options.Ninits = 1;
options.zeromean = 0;
[options, data] = checkoptions(options, X, T, 0);
options.nu = sum(T)/200;
options.Gamma = em_init(data,T,options,options.Sind);
hmm_wr = struct('train',struct());
hmm_wr.K = options.K;
hmm_wr.train = options;
hmm_wr=hmmhsinit(hmm_wr);
[hmm_wr,residuals_wr]=obsinit(data,T,hmm_wr,options.Gamma);
hmm_wr.P = P;
hmm_wr.Pi = Pi_0; 

% compare to matlab code
display('Pure matlab')
tic;
for i=1:400,
[g_check, ~, Xi_check, ~, scale_check, B] = hsinference(data, T, hmm_wr, [], options);
end
toc

% run mex file
display('Mex implementation')
tic;
for i=1:400,
[gamma, Xi, scale] = hidden_state_inference_mx(B, Pi_0, P, options.order);
end
toc

% check results match
display('Differences in result:')
display(max(abs(g_check(:) - gamma(:))));
display(max(abs(Xi(:) - Xi_check(:))));
display(max(abs(scale(:) - scale_check(:))));

fprintf('Machine precision is about %g. \n', eps(max(abs(gamma(:)))));

% check probability bounds
assert(max(gamma(:))<=1 & min(gamma(:))>=0)
assert(max(Xi(:))<=1 & min(Xi(:))>=0);



