function [mcv,cv] = cvhmmmar (data,T,options)
%
% Obtains the cross-validated sum of prediction quadratic errors.
%
% INPUT
% data          observations, either a struct with X (time series) and C (classes, optional)
%                             or just a matrix containing the time series
% T             length of series
% options       structure with the training options - see documentation
%
% OUTPUT
% mcv      the averaged cross-validated likelihood and/or fractional mean squared error
% cv       the averaged cross-validated likelihood and/or fractional mean squared error per fold
%
% Author: Diego Vidaurre, OHBA, University of Oxford


[options,data] = checkoptions(options,data,T,1);

if length(options.embeddedlags)>1
    X = []; C = [];
    for in=1:length(T)
        [x, ind] = embedx(data.X(sum(T(1:in-1))+1:sum(T(1:in)),:),options.embeddedlags); X = [X; x ];
        c = data.C( sum(T(1:in-1))+1: sum(T(1:in)) , : ); c = c(ind,:); C = [C; c];
        T(in) = size(c,1);
    end
    data.X = X; data.C = C;
end

options.verbose = options.cvverbose;
options.updateGamma = options.K>1;

mcv = 0; if options.cvmode>2, mcv = [0 0]; end
if length(options.cvfolds)==1,
    options.cvfolds = crossvalind('Kfold', length(T), options.cvfolds);
end
nfolds = max(options.cvfolds);

[orders,order] = formorders(options.order,options.orderoffset,options.timelag,options.exptimelag);
Sind = formindexes(orders,options.S);
%if ~options.zeromean, Sind = [true(1,size(Sind,2)); Sind]; end

Ttotal = 0;
cv = zeros(nfolds,1);
for fold=1:nfolds
    datatr.X = []; datatr.C = []; Ttr = [];
    datate.X = []; datate.C = []; Tte = []; Gammate = []; test = [];
    % build fold
    for i=1:length(T)
        t0 = sum(T(1:(i-1)))+1; t1 = sum(T(1:i)); Ti = t1-t0+1;
        if options.cvfolds(i)==fold % in testing
            datate.X = [datate.X; data.X(t0:t1,:)];
            datate.C = [datate.C; data.C(t0:t1,:)];
            Tte = [Tte Ti];
            Gammate = [Gammate; data.C(t0+order:t1,:)];
            test = [test; ones(Ti,1)];
        else % in training
            datatr.X = [datatr.X; data.X(t0:t1,:)];
            datatr.C = [datatr.C; data.C(t0:t1,:)];
            Ttr = [Ttr Ti];
        end
    end
    Ttotal = Ttotal + sum(Tte) - length(Tte)*order;
    
    %if options.whitening>0
    %    mu = mean(datatr.X);
    %    datatr.X = bsxfun(@minus, datatr.X, mu);
    %    datate.X = bsxfun(@minus, datate.X, mu);
    %    [V,D] = svd(datatr.X'*datatr.X);
    %    A = sqrt(size(datatr.X,1)-1)*V*sqrtm(inv(D + eye(size(D))*0.00001))*V';
    %    datatr.X = datatr.X*A;
    %    datate.X = datate.X*A;
    %    %iA = pinv(A);
    %end
    
    Fe = Inf;
    for it=1:options.cvrep
        if options.verbose, fprintf('CV fold %d, repetition %d \n',fold,it); end
        % init Gamma
        options.Gamma = [];
        if options.K > 1
            if options.initrep>0 && strcmp(options.inittype,'HMM-MAR')
                options.Gamma = hmmmar_init(datatr,Ttr,options,Sind);
            elseif options.initrep>0 && strcmp(options.inittype,'EM')
                options.nu = sum(T)/200;
                options.Gamma = em_init(datatr,Ttr,options,Sind);
            else
                options.Gamma = [];
                for in=1:length(Ttr)
                    gamma = rand(Ttr(in)-options.maxorder,options.K);
                    options.Gamma = [options.Gamma; gamma ./ repmat(sum(gamma,2),1,options.K)];
                end
            end
        end
        % train
        hmmtr=struct('train',struct());
        hmmtr.K = options.K; 
        hmmtr.train = options; 
        hmmtr.train.Sind = Sind; 
        hmmtr=hmmhsinit(hmmtr);
        [hmmtr,residualstr,W0tr]=obsinit(datatr,Ttr,hmmtr,options.Gamma);
        [hmmtr,~,~,fe,actstates] = hmmtrain(datatr,Ttr,hmmtr,options.Gamma,residualstr); fe = fe(end);
        % test
        if fe<Fe,
            Fe = fe;
            residualste =  getresiduals(datate.X,Tte,hmmtr.train.Sind,hmmtr.train.order,hmmtr.train.maxorder,...
                hmmtr.train.orderoffset,hmmtr.train.timelag,hmmtr.train.exptimelag,hmmtr.train.zeromean,W0tr);
            if options.cvmode==1
                [~,~,~,LL] = hsinference(datate,Tte,hmmtr,residualste);
                cv(fold) = sum(LL);
            elseif options.cvmode==2
                [~,fracerr] = hmmerror(datate.X,Tte,hmmtr,Gammate,test,residualste,actstates);
                cv(fold) = mean(fracerr);
            else
                [~,~,~,LL] = hsinference(datate,Tte,hmmtr,residualste);
                cv(fold,1) = sum(LL);
                [~,fracerr] = hmmerror(datate.X,Tte,hmmtr,Gammate,test,residualste,actstates);
                cv(fold,2) = mean(fracerr);
            end
        end
    end
    if options.cvmode==1, mcv = mcv + cv(fold);
    elseif options.cvmode==2, mcv = mcv + (sum(Tte) - length(Tte)*options.maxorder) * cv(fold);
    else mcv(1) = mcv(1) + cv(fold,1); mcv(2) = mcv(2) + (sum(Tte) - length(Tte)*options.maxorder) * cv(fold,2);
    end
end
if options.cvmode==2, mcv = mcv / Ttotal;
elseif options.cvmode==3, mcv(2) = mcv(2) / Ttotal;
end

end
