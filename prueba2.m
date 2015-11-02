clear all
addpath(genpath('.'))
load ../MarkDebug/data
load ../MarkDebug/GammaInit.mat

options.K = 8;
options.cyc = 100;
options.tol = -100;
options.order = 4; 
options.orderoffset = 0; 
options.timelag =1; 
options.exptimelag =0;

options.covtype = 'diag';
options.order = 2;
options.zeromean = 0;
options.symmetricprior = 1;

[hmm, Gamma, ~, ~, ~, ~, fehist] = hmmmar(data,size(data,1),options);