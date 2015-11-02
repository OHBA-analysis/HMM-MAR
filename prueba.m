addpath(genpath('.'))
load ../HMM-MAR-OSL/prueba.mat
load ../hmmbox_4_1_OSL/GammaInit.mat
%load GammaInit.mat

TT = {4000,[1000 1000 1000 1000]};
Matcov = {'diag','uniquediag','full','uniquefull'};
Order = [0 2];
Zeromean = [0 1];
Symmetric = [0 1];

options.cyc = 500;
options.tol = -100;
options.order = 0; options.orderoffset = 0; options.timelag =0; options.exptimelag =0;

FE = [];

for it = 1:length(TT)
    T = TT{it};
    for imatcov = 1:length(Matcov)
        matcov = Matcov{imatcov};
        options.covtype = matcov;
        for order = Order
            options.order = order;
            options.Gamma = GammaInit(order*length(T)+1:end,:);
            if order==0
                options.orderoffset = 0; options.timelag =0;
            else
                options.orderoffset = 1; options.timelag =1;
            end
            for zeromean= Zeromean
                options.zeromean=zeromean;
                options.symmetricprior = 0;
                fprintf('%d %s %d %d 0 \n',it, matcov, order, zeromean)
                [hmm, ~, ~, ~, ~, ~, fehist] = hmmmar(X,T,options);
                FE = [FE; fehist'];
%                 options.symmetricprior = 1;
%                 fprintf('%d %s %d %d 1 \n',it, matcov, order, zeromean)
%                 [hmm, ~, ~, ~, ~, ~, fehist] = hmmmar(X,T,options);
%                 FE = [FE; fehist'];
            end
        end
    end
    
end
    
%save('/tmp/FE.mat','FE')

%% Different configurations per state

addpath(genpath('.'))
load ../HMM-MAR-OSL/prueba.mat
load ../hmmbox_4_1_OSL/GammaInit.mat
%load GammaInit.mat

T = [1000 1000 1000 1000];

clear options
options.cyc = 150;
options.tol = -100;

options.K = 3; 
options.order = 2; %baseline
options.orderoffset = 0; 
options.timelag = 1; 
options.exptimelag = 0;
options.zeromean = 1; 
options.symmetric = 1; 
options.covtype = 'diag';
options.uniqueAR = 1;
%options.Gamma = rand(4000-2*4,3); options.Gamma = options.Gamma ./ repmat(sum(options.Gamma,2),1,options.K);

options.state(1) = struct('train',[]);

options.state(2) = struct('train',struct());
options.state(2).train.order = 2;  
options.state(2).train.orderoffset = 0; 
options.state(2).train.timelag = 1; 
options.state(2).train.exptimelag = 0;
options.state(2).train.zeromean = 1; 
options.state(2).train.symmetric = 1; 
options.state(2).train.covtype = 'diag';
options.state(2).train.uniqueAR = 1;
priorS = randn(size(X,2) + ~options.state(2).train.zeromean); priorS = priorS' * priorS;
priormu = randn(size(X,2) + ~options.state(2).train.zeromean,1);
options.state(2).train.prior.S = priorS;
options.state(2).train.prior.Mu = priormu;

options.state(3) = struct('train',struct());
options.state(3).train.order = 2;  
options.state(3).train.orderoffset = 0; 
options.state(3).train.timelag = 1; 
options.state(3).train.exptimelag = 0;
options.state(3).train.zeromean = 1; 
options.state(3).train.symmetric = 1; 
options.state(3).train.covtype = 'diag';
options.state(3).train.uniqueAR = 1;
% 
[hmm, Gam, ~, ~, ~, ~, fehist] = hmmmar(X,T,options);
                
%save('/tmp/FE.mat','FE')

%% Cross validation

options.cyc = 5;
options.initcyc = 1; options.initrep = 1; 
options.tol = 1e-6;
options.Gamma = [];

T = [1000 1000 1000 1000];
MCV = [];

for imatcov = 1:length(Matcov)
    matcov = Matcov{imatcov};
    options.covtype = matcov;
    for order = Order
        options.order = order;
        if order==0
            options.orderoffset = 0; options.timelag =0;
        else
            options.orderoffset = 1; options.timelag =1;
        end
        for zeromean=Zeromean
            if zeromean==1 && order==0, continue; end
            options.zeromean=zeromean;
            fprintf('%s %d %d \n',matcov, order, zeromean)
            mcv = cvhmmmar (X,T,options);
            MCV = [MCV; mcv];
        end
    end
end
    

%%

load /tmp/FE.mat

diffFE = (FE(:,1:end-1) - FE(:,2:end)) ./ repmat(FE(:,1) - FE(:,end) , 1 ,size(FE,2)-1); 
diffFE(diffFE>0) = 0; 
imagesc(diffFE); colorbar

c = 1; 
for it = 2
    T = TT{it};
    for imatcov = 1:length(Matcov)
        matcov = Matcov{imatcov};
        for order = Order
            for zeromean=Zeromean
                fprintf('%d: %d %s %d %d: 0 %g  \n',c,it, matcov, order, zeromean,min(diffFE(c,:))); c = c+1;  
                fprintf('%d: %d %s %d %d: 1 %g  \n',c,it, matcov, order, zeromean,min(diffFE(c,:))); c = c+1;  
            end
        end 
    end
    
end

%% Test spectra

options.cyc = 10;
options.order = 4; options.timelag=1; options.orderoffset = 1;

for it = 1:2
    T = TT{it};
    for imatcov = 1:length(Matcov)
        for zeromean=Zeromean
            matcov = Matcov{imatcov};
            options.covtype = matcov;
            options.zeromean = zeromean;
            [hmm, Gammahat, ] = hmmmar(X,T,options);
            params = struct('Fs',200);
            params.fpass = [1 48];
            params.tapers = [4 7];
            params.p = 0.05;
            params.win = 500;
            params.Gamma = Gammahat;
            X2 = []; T2 = T;
            order = (sum(T) - size(Gammahat,1)) / length(T);
            for in=1:length(T)
                t0 = sum(T(1:in-1)) - order*(in-1) + 1; t1 = sum(T(1:in)) - order*in;
                T2(in) = T(in) - order;
                X2 = [X2; X(t0:t1,:)];
            end
            if length(T)>1,
                params.p = 0.05;
            else
                params.p = 0;
            end
            fitmar = hmmspectramar(hmm,X,T,params);
            fitmt = hmmspectramt(X2,T2,params);
        end
    end
end

        
