% The idea of this script is that everytime someone changes something in
% the core code, runs it to check everything is still alright. 
% That is: we won't update stuff to git without running this script first

% gen data
X = randn(4000,3); T = 500*ones(8,1); 
B = rand(3);  
for i=1:length(T)
    X((1:250)+sum(T(1:i-1)),:) = X((1:250)+sum(T(1:i-1)),:) * B; 
    for j=1:3
        X((1:500)+sum(T(1:i-1)),j) = smooth(X((1:500)+sum(T(1:i-1)),j),10);
    end
end
C = rand(3,10); 
X = X * C;
extreme = find(X(:) > 5*std(X(:)));

%%

% standard inference, hmmmar init
options = struct();
options.K = 2; 
options.Fs = 200; 
options.pca = 0.95;
options.varimax = 0;
options.filter = [];
options.detrend = 1; 
options.downsample = 100; 
options.inittype = 'hmmmar';
options.standardise = 1;
options.standardise_pc = 1;
options.cyc = 5;
options.initcyc = 2;
options.tol = 1e-7;
options.initrep = 2;
options.verbose = 0; 
options.useParallel = 0; 
%options.grouping = [1 1 1 1 1 1 1 1]; 
%options.grouping = [1 1 1 1 2 2 2 2];

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
            [hmm,Gamma,Xi,vpath,~,~,fe] = hmmmar(X,T,options); 
            if isfield(options,'grouping')
                fe2 = hmmfe(X,T,hmm,[],[],[],options.grouping); %(fe(end)-fe2)/fe2
            else
                fe2 = hmmfe(X,T,hmm); %(fe(end)-fe2)/fe2
            end
            fitmt = hmmspectramt(X,T,Gamma,options);
            if order > 0
                fitmar1 = hmmspectramar([],[],hmm);
                fitmar2 = hmmspectramar(X,T,[],Gamma,options);
            end
        end
    end
end


% One channel
options.order = 2;
options.pca = 0; 
options.covtype = 'diag'; 
options.embeddedlags = 0;
[hmm,Gamma,Xi,vpath,~,~,fe] = hmmmar(X(:,1),T,options);
fitmt = hmmspectramt(X(:,1),T,Gamma,options);
fitmar1 = hmmspectramar([],[],hmm);
fitmar2 = hmmspectramar(X(:,1),T,[],Gamma,options);
if isfield(options,'grouping')
    fe2 = hmmfe(X(:,1),T,hmm,[],[],[],options.grouping); %(fe(end)-fe2)/fe2
else
    fe2 = hmmfe(X(:,1),T,hmm); %(fe(end)-fe2)/fe2
end


% Embedded HMM
options.order = 0;
options.zeromean = 1;
options.covtype = 'full'; 
options.embeddedlags = -2:2;
options.pca = 6;
[hmm,Gamma,Xi,vpath,~,~,fe] = hmmmar(X,T,options);
fitmt = hmmspectramt(X,T,Gamma,options);

if isfield(options,'grouping')
    fe2 = hmmfe(X,T,hmm,[],[],[],options.grouping); %(fe(end)-fe2)/fe2
else
    fe2 = hmmfe(X,T,hmm); %(fe(end)-fe2)/fe2
end

options.embeddedlags = 0; 

% random initialization
options.inittype = 'random';
[hmm,Gamma,Xi,vpath,~,~,fe] = hmmmar(X,T,options);
if isfield(options,'grouping')
    fe2 = hmmfe(X,T,hmm,[],[],[],options.grouping); %(fe(end)-fe2)/fe2
else
    fe2 = hmmfe(X,T,hmm); %(fe(end)-fe2)/fe2
end

% Specifying data.C (semisupervised)
data = struct(); 
data.X = X;
data.C = nan(length(X),2);
for i=1:4
    data.C((200:250)+sum(T(1:i-1)),1) = 0;
    data.C((200:250)+sum(T(1:i-1)),2) = 1;
end
options.downsample = 0; 

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
            [hmm,Gamma,Xi,vpath,~,~,fe] = hmmmar(data,T,options);
            if isfield(options,'grouping')
                fe2 = hmmfe(X,T,hmm,[],[],[],options.grouping); %(fe(end)-fe2)/fe2
            else
                fe2 = hmmfe(X,T,hmm); %(fe(end)-fe2)/fe2
            end
        end
    end
end

% Embedded HMM
options.order = 0;
options.zeromean = 1;
options.covtype = 'full'; 
options.embeddedlags = -2:2;
options.pca = 6; 
[hmm,Gamma,Xi,vpath,~,~,fe] = hmmmar(X,T,options);
if isfield(options,'grouping')
    fe2 = hmmfe(X,T,hmm,[],[],[],options.grouping); %(fe(end)-fe2)/fe2
else
    fe2 = hmmfe(X,T,hmm); %(fe(end)-fe2)/fe2
end
%options.embeddedlags = 0; 

% random initialization
options.inittype = 'random';
[hmm,Gamma,Xi,vpath,~,~,fe] = hmmmar(X,T,options);
if isfield(options,'grouping')
    fe2 = hmmfe(X,T,hmm,[],[],[],options.grouping); %(fe(end)-fe2)/fe2
else
    fe2 = hmmfe(X,T,hmm); %(fe(end)-fe2)/fe2
end


%% stochastic inference

disp('*** STOCHASTIC LEARNING***')

options = struct();
options.K = 2; 
options.Fs = 200; 
options.standardise = 1;
options.pca = 2;
options.varimax = 0; 
options.filter = [];
options.detrend = 1; 
options.downsample = 100; 
options.verbose = 0;
options.cyc = 5;
options.initcyc = 2;
options.initrep = 2;
options.tol = 1e-7;
options.useParallel = 0;
options.grouping = [1 1 1 1 1 1 1 1];
%options.grouping = [1 1 1 1 2 2 2 2];

options.BIGNbatch = 3;
options.BIGNinitbatch = 3;
options.BIGtol = 1e-7;
options.BIGcyc = 5;
options.BIGundertol_tostop = 5;
options.BIGdelay = 1; 
options.BIGforgetrate = 0.7;
options.BIGbase_weights = 0.9;
    
for covtype = {'full','diag'} %,'uniquefull','uniquediag'}
    for order = [0 2]
        for zeromean = [0 1]
            if (strcmp(covtype,'uniquefull') || strcmp(covtype,'uniquediag')) ...
                    && zeromean==1 && order==0
                continue;
            end
            options.covtype = covtype{1};
            options.order = order;
            options.zeromean = zeromean;
            [hmm,Gamma,~,vpath,~,~,fe] = hmmmar(X,T,options);
            [~,Xi] = hmmdecode(X,T,hmm);
            if isfield(options,'grouping')
                fe2 = hmmfe(X,T,hmm,[],[],[],options.grouping); %(fe(end)-fe2)/fe2
            else
                fe2 = hmmfe(X,T,hmm); %(fe(end)-fe2)/fe2
            end
            fitmt = hmmspectramt(X,T,Gamma,options);
            if isnan((fe(end)-fe2)/fe2), error('NaN'); end
        end
    end
end

% One channel
options.order = 2;
options.pca = 0; 
options.covtype = 'diag'; 
options.embeddedlags = 0;
[hmm,Gamma,~,vpath,~,~,fe] = hmmmar(X(:,1),T,options);
[~,Xi] = hmmdecode(X(:,1),T,hmm);
fitmt = hmmspectramt(X(:,1),T,Gamma,options);
if isfield(options,'grouping')
    fe2 = hmmfe(X(:,1),T,hmm,[],[],[],options.grouping); %(fe(end)-fe2)/fe2
else
    fe2 = hmmfe(X(:,1),T,hmm); %(fe(end)-fe2)/fe2
end
if isnan((fe(end)-fe2)/fe2), error('NaN'); end

% Using S
options.S = - ones(10);
options.S(6:10,1:5) = ones(5) - 2*eye(5);
options.order = 2;
options.pca = 0; 
options.covtype = 'diag'; 
options.embeddedlags = 0;
options.zeromean = 1; 
[hmm,Gamma,Xi,vpath,~,~,fe] = hmmmar(X,T,options);
fitmt = hmmspectramt(X,T,Gamma,options);
options = rmfield(options,'S');

% Embedded HMM
options.order = 0;
options.zeromean = 1;
options.covtype = 'full';
options.embeddedlags = -2:2;
options.pca = 3;
[hmm,Gamma,Xi,vpath,~,~,fe] = hmmmar(X,T,options);
[~,Xi] = hmmdecode(X,T,hmm);
fitmt = hmmspectramt(X,T,Gamma,options);
if isfield(options,'grouping')
    fe2 = hmmfe(X,T,hmm,[],[],[],options.grouping); %(fe(end)-fe2)/fe2
else
    fe2 = hmmfe(X,T,hmm); %(fe(end)-fe2)/fe2
end
if isnan((fe(end)-fe2)/fe2), error('NaN'); end
