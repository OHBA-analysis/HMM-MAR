% The idea of this script is that everytime someone changes something in
% the core code, runs it to check everything is still alright. 
% That is: we won't update stuff to git without running this script first

% gen data
X = randn(2000,3); T = 250*ones(8,1); 
B = rand(3);
for i=1:4, 
    X((1:100)+sum(T(1:i-1)),:) = X((1:100)+sum(T(1:i-1)),:) * B; 
    for j=1:3
        X((1:250)+sum(T(1:i-1)),j) = smooth(X((1:250)+sum(T(1:i-1)),j),10);
    end
end

%%

% standard inference, GMM init
options = struct();
options.K = 2; 
options.tol = 1e-7;
options.cyc = 12;
options.inittype = 'GMM';
options.DirichletDiag = 10;
options.initcyc = 4;
options.initrep = 2;
options.standardise = 1;
options.verbose = 0; 
options.useParallel = 0; 

for covtype = {'full','diag','uniquefull','uniquediag'}
    for order = [0 2]
        for zeromean = [0 1]
            if (strcmp(covtype,'uniquefull') || strcmp(covtype,'uniquediag')) ...
                    && zeromean==1 && order==0
                continue;
            end
            options.covtype = covtype{1};
            options.order = order;
            options.zeromean = zeromean;
            [hmm,Gamma] = hmmmar(X,T,options);
        end
    end
end

% HMM-MAR initialization
options.inittype = 'hmmmar';
[hmm,Gamma] = hmmmar(X,T,options);


%% stochastic inference
options = struct();
options.K = 2; 
options.tol = 1e-7;
options.cyc = 12;
options.inittype = 'GMM';
options.DirichletDiag = 10;
options.initcyc = 4;
options.initrep = 2;
options.standardise = 1;
options.verbose = 0; 
options.useParallel = 0;
%options.AR = 1;

options.BIGNbatch = 3;
options.BIGNinitbatch = 3;
options.BIGtol = 1e-7;
options.BIGcyc = 10;
options.BIGundertol_tostop = 5;
options.BIGdelay = 5; 
options.BIGforgetrate = 0.7;
options.BIGbase_weights = 0.9;
    
for covtype = {'full','diag'} %,'uniquefull','uniquediag'}
    if strcmp(covtype,'full') && options.AR==1, continue; end
    for order = [0 2]
        for zeromean = [0 1]
            if (strcmp(covtype,'uniquefull') || strcmp(covtype,'uniquediag')) ...
                    && zeromean==1 && order==0
                continue;
            end
            options.covtype = covtype{1};
            options.order = order;
            options.zeromean = zeromean;
            [hmm,Gamma] = hmmmar(X,T,options);
        end
    end
end

%% Test stochastic with K=1
options.K = 1;
[hmm,Gamma] = hmmmar(X,T,options);

% you need to test the other options of initialisation as well

