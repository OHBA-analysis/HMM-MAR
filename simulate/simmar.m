function [X,options] = simmar(data,T,Tnew,options)
%
% Simulate data from one single MAR state, which parameters will be
% estimated from the data
%
% INPUTS:
%
% data                  observations, a matrix containing the time series
% T                     Number of time points for each time series
% Tnew                  Length of new time series
% options               Options for the MAR and preprocessing
% options.AR	 	If 1, one AR model will be fitter to each channel
%			with no cross-term interactions
%
% OUTPUTS
% X                     simulated observations
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2017)

N = length(T); Nnew = length(Tnew);
if isstruct(data), data = data.X; end
ndim = size(data,2);

% Check data
if iscell(T)
    for i = 1:length(T)
        if size(T{i},1)==1, T{i} = T{i}'; end
    end
    if size(T,1)==1, T = T'; end
    T = cell2mat(T);
end
checkdatacell;

% Check options
if isfield(options,'AR'), options.AR = 0; end
options = checkspelling(options);
options.K = 1; options.updateGamma = 0; options.updateP = 0;
if ~isfield(options,'order')
    error('In order to sample data from a MAR model, options.order is needed')
end
options.DirichletDiag = 100; % this is not used, it's just to stop the warning message
[options,data] = checkoptions(options,data,T,0);
if length(options.embeddedlags) > 1
    error('It is not currently possible to generate data with options.embeddedlags ~= 0');
end

% Standardise data and control for ackward trials
data = standardisedata(data,T,options.standardise);
% Filtering
if ~isempty(options.filter)
    data = filterdata(data,T,options.Fs,options.filter);
end
% Detrend data
if options.detrend
    data = detrenddata(data,T);
end
% Leakage correction
if options.leakagecorr ~= 0
    data = leakcorr(data,T,options.leakagecorr);
end
% Hilbert envelope
if options.onpower
    data = rawsignal2power(data,T);
end
% Downsampling
if options.downsample > 0
    [data,T] = downsampledata(data,T,options.downsample,options.Fs);
end
data = data.X;

if options.order > 0
    orders = formorders(options.order,options.orderoffset,...
        options.timelag,options.exptimelag);
    maxorder = max(orders);
    XX = formautoregr(data,T,orders,maxorder,options.zeromean,0);
    Y =  getresiduals(data,T,1,max(orders));
    if isfield(options,'AR') && options.AR
        for j = 1:ndim
            ind = (0:options.order-1)*ndim + j;
            W(ind,j) = pinv(XX(:,ind)) * Y(:,j);
        end
    else
        W = pinv(XX) * Y;
    end
    R = Y - XX * W;
    mu = mean(R);
    C = cov(R);
    nz = (~options.zeromean);
else
    mu = mean(data);
    C = cov(data);
end

if options.order > 0 % a MAR is generating the data
    d = 500;
    X = zeros(sum(Tnew),ndim);
    for n = 1:Nnew
        Xin = mvnrnd(repmat(mu,d+Tnew(n),1),C);
        for t=maxorder+1:Tnew(n)+d
            XX = ones(1,length(orders)*ndim+nz);
            for i=1:length(orders)
                o = orders(i);
                XX(1,(1:ndim) + (i-1)*ndim + nz) = Xin(t-o,:);
            end
            Xin(t,:) = Xin(t,:) + XX * W;
        end
        ind = (1:Tnew(n)) + sum(Tnew(1:n-1));
        X(ind,:) = Xin(d+1:end,:);
    end
else
    X = mvnrnd(repmat(mu,sum(Tnew),1),C);
end

end

