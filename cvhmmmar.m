function [mcv,cv] = cvhmmmar(data,T,options)
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
if ~all(options.grouping==1)
    warning('grouping option is not yet implemented in cvhmmmar')
    options.grouping = ones(length(T),1); 
end
    
options.verbose = options.cvverbose;
options.dropstates = 0;
options.updateGamma = options.K>1;

if length(options.embeddedlags)>1
    X = []; C = [];
    for in=1:length(T)
        [x, ind] = embedx(data.X(sum(T(1:in-1))+1:sum(T(1:in)),:),options.embeddedlags); X = [X; x ];
        c = data.C( sum(T(1:in-1))+1: sum(T(1:in)) , : ); c = c(ind,:); C = [C; c];
        T(in) = size(c,1);
    end
    data.X = X; data.C = C;
end

mcv = 0; if options.cvmode>2, mcv = [0 0]; end
if length(options.cvfolds)==1
    options.cvfolds = crossvalind('Kfold', length(T), options.cvfolds);
end
nfolds = max(options.cvfolds);
[orders,order] = formorders(options.order,options.orderoffset,options.timelag,options.exptimelag);
Sind = formindexes(orders,options.S);
if ~options.zeromean, Sind = [true(1,size(Sind,2)); Sind]; end
maxorder = options.maxorder;
W0tr = [];
Ttotal = 0;
cv = zeros(nfolds,1);

for fold=1:nfolds
    
    Ttr = [];
    indtr1 = []; indtr2 = [];
    Tte = []; 
    indte1 = []; indte2 = [];
    test = [];
    % build fold
    for i=1:length(T)
        t0 = sum(T(1:(i-1)))+1; t1 = sum(T(1:i)); 
        s0 = sum(T(1:(i-1)))-order*(i-1)+1; s1 = sum(T(1:i))-order*i;
        Ti = t1-t0+1;
        if options.cvfolds(i)==fold % in testing
            indte1 = [indte1 (t0:t1)];
            indte2 = [indte2 (s0:s1)];
            Tte = [Tte Ti];
            test = [test; ones(Ti,1)];
        else % in training
            indtr1 = [indtr1 (t0:t1)];
            indtr2 = [indtr2 (s0:s1)];
            Ttr = [Ttr Ti];
        end
    end
    datatr.X = data.X(indtr1,:); datatr.C = data.C(indtr2,:);
    datate.X = data.X(indte1,:); datate.C = data.C(indte2,:);
    Gammate = data.C(indte2,:);
    
    Ttotal = Ttotal + sum(Tte) - length(Tte)*order;
    
    Fe = Inf;
      
    for it=1:options.cvrep
        if options.verbose, fprintf('CV fold %d, repetition %d \n',fold,it); end

        if isfield(options,'orders')
            options = rmfield(options,'orders');
        end
        if isfield(options,'maxorder')
            options = rmfield(options,'maxorder');
        end
        [hmmtr,~,~,~,~,~,fe] = hmmmar (datatr,Ttr,options); fe = fe(end);
        hmmtr.train.Sind = Sind;
        hmmtr.train.maxorder = maxorder;
               
        % test
        if fe<Fe
            Fe = fe;
            residualste =  getresiduals(datate.X,Tte,hmmtr.train.Sind,hmmtr.train.maxorder,hmmtr.train.order,...
                hmmtr.train.orderoffset,hmmtr.train.timelag,hmmtr.train.exptimelag,hmmtr.train.zeromean,W0tr);
            if options.cvmode==1
                [~,~,~,LL] = hsinference(datate,Tte,hmmtr,residualste);
                cv(fold) = sum(LL);
            elseif options.cvmode==2
                [~,fracerr] = hmmerror(datate.X,Tte,hmmtr,Gammate,test,residualste);
                cv(fold) = mean(fracerr);
            else
                [~,~,~,LL] = hsinference(datate,Tte,hmmtr,residualste);
                cv(fold,1) = sum(LL);
                [~,fracerr] = hmmerror(datate.X,Tte,hmmtr,Gammate,test,residualste);
                cv(fold,2) = mean(fracerr);
            end
        end
    end
    if options.cvmode==1, mcv = mcv + cv(fold);
    elseif options.cvmode==2, mcv = mcv + (sum(Tte) - length(Tte)*maxorder) * cv(fold);
    else mcv(1) = mcv(1) + cv(fold,1); mcv(2) = mcv(2) + (sum(Tte) - length(Tte)*maxorder) * cv(fold,2);
    end
    
end

if options.cvmode==2, mcv = mcv / Ttotal;
elseif options.cvmode==3, mcv(2) = mcv(2) / Ttotal;
end

end
