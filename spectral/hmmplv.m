function [plv,powercorr] = hmmplv(data, T, Gamma, options)
% Computes the state-wise Phase Locking Value (PLV) for (time X numChannels) data
%
% Input parameters:
%   data is a numTimePoints x numChannels x numTimePoints data matrix
%   options must contain
%       .Fs is the sampling rate
%       .fpass specifies the limits of the frequency
%               (e.g. .fpass = [13 30] for beta band) 
%       .order specifies the order of the FIR filter
%   One can set it to include about 4 to 5 cycles of the chosen frequency;
%   for example, .order = 50 for data sampled at
%     500Hz corresponds to 100ms and contains ~4 cycles of gamma band (40 Hz).
%
% Output parameters:
%   plv is (numChannels x numChannels x nSegments x nStates)
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2017)

if nargin<4
    error('You need to provided an options struct, with fields Fs, fpass and order')
end

if iscell(data)
    if ~iscell(T), error('If you provide data as a cell, T must be a cell too'); end
    X = loadfile_plv(data{1},T{1});
    ndim = size(X,2);
    TT = [];
    for j=1:length(data)
        if size(T{j},1)==1, T{j} = T{j}'; end
        TT = [TT; T{j}];
    end
    if isempty(Gamma)
        Gamma = ones(sum(TT),1);
    end
    order = (sum(TT) - size(Gamma,1)) / length(TT);
    TT = TT - order;
else
    ndim = size(data,2); T = double(T); 
    if isempty(Gamma)
        Gamma = ones(sum(T),1);
    end
    order = (sum(T) - size(Gamma,1)) / length(T);
    % remove the exceeding part of X (with no attached Gamma)
    if order>0
        data2 = zeros(sum(T)-length(T)*order,ndim);
        for in = 1:length(T)
            t0 = sum(T(1:in-1)); t00 = sum(T(1:in-1)) - (in-1)*order;
            data2(t00+1:t00+T(in)-order,:) = data(t0+1+order:t0+T(in),:);
        end
        T = T - order;
        data = data2; clear data2;
    end
    TT = T;
end

K = size(Gamma,2);

if size(Gamma,2)==1 && ~all(Gamma==1) % viterbi path
    vpath = Gamma;
    Gamma = zeros(size(Gamma,1),K);
    for k=1:K, Gamma(vpath==k,k) = 1; end
end

N = length(T);
Fs = options.Fs;
filtPts = fir1(options.order, 2/Fs*options.fpass);
discard = 2*round(Fs/2);

plv = zeros(ndim, ndim, N, K);
powercorr = zeros(ndim, ndim, N, K);

c = 0;
for j = 1:N
    if iscell(data)
        X = loadfile_plv(data{j},T{j}); 
        Tj = T{j};
    else
        X = data((1:TT(j)) + sum(TT(1:j-1)) , : ); 
        Tj = TT(j);        
    end
    lenTj = length(Tj);
    G = Gamma(c + (1:sum(Tj)-lenTj*order) , :); c = c + size(G,1);
    phase = zeros(sum(Tj)-lenTj*order-lenTj*discard,ndim);
    power = zeros(sum(Tj)-lenTj*order-lenTj*discard,ndim);
    Gammaj = zeros(sum(Tj)-lenTj*order-lenTj*discard,K);
    for i = 1:length(Tj)
        t0 = sum(Tj(1:i-1)); 
        Xij = filter(filtPts, 1, X(t0+1+order:t0+Tj(i),:) , [], 1);
        range = (round(Fs/2)+1) : (size(Xij,1)-round(Fs/2)); % discard some data
        t0 = sum(Tj(1:i-1)) - (i-1)*order - (i-1)*lenTj;
        for n = 1:ndim
            h = hilbert(Xij(:,n));
            s = angle(h);
            phase((t0+1):(t0+Tj(i)-order-discard),n) = s(range);
            s = abs(h);
            power((t0+1):(t0+Tj(i)-order-discard),n) = s(range);
        end
        t0g = sum(Tj(1:i-1)) - (i-1)*order;
        g = G((t0g+1):(t0g+Tj(i)-order),:);
        range = (round(Fs/2)+1) : (size(g,1)-round(Fs/2)); % discard some data
        Gammaj((t0+1):(t0+Tj(i)-order-discard),:) = g(range,:);
    end
    
    for n1 = 1:ndim-1
        for n2 = n1+1:ndim
            for k=1:K
                plv(n1,n2,j,k) = abs(sum(Gammaj(:,k) .* exp(1i*(phase(:,n1)-phase(:,n2))))) ...
                    / sum(Gammaj(:,k));
                plv(n2,n1,j,k) = plv(n1,n2,j,k);
                m1 = sum(Gammaj(:,k) .* power(:,n1)) / sum((Gammaj(:,k)));
                m2 = sum(Gammaj(:,k) .* power(:,n2)) / sum((Gammaj(:,k)));
                s1 = sum(Gammaj(:,k) .* (power(:,n1) - m1).^2) / sum((Gammaj(:,k)));
                s2 = sum(Gammaj(:,k) .* (power(:,n2) - m2).^2) / sum((Gammaj(:,k)));
                s12 = sum(Gammaj(:,k) .* ...
                    ((power(:,n1) - m1).^2) .* ((power(:,n2) - m2).^2)) ./ ...
                    sum((Gammaj(:,k)));
                powercorr(n1,n2,j,k) = s12 / sqrt(s1) / sqrt(s2);
                powercorr(n2,n1,j,k) = powercorr(n1,n2,j,k);
            end
        end
    end
end

if size(plv,3)==1, plv = permute(plv,[1 2 4 3]); end

end

function X = loadfile_plv(f,T)
if ischar(f)
    if ~isempty(strfind(f,'.mat')), load(f,'X');
    else X = dlmread(f);
    end
else
    X = f;
end
for i=1:length(T)
    t = (1:T(i)) + sum(T(1:i-1));
    X(t,:) = X(t,:) - repmat(mean(X(t,:)),length(t),1);
    X(t,:) = X(t,:) ./ repmat(std(X(t,:)),length(t),1);
end
end