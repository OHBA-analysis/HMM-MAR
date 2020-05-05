function D = computeDistMatrix_AVFC (data,T,options)
%
% It uses KL-divergence to compute a subject-by-subject distance matrix
% of average FC matrices (on fMRI). 
%
% INPUT
% data          observations; in this case it has to be a cell, each with
%               the data for one subject
% T             length of series, also a cell. 
% 
% OUTPUT
% D             (N by N) distance matrix, with the distance between each
%               pair of subjects in "HMM space"
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2020)

if ~iscell(data) || ~iscell(T), error('X and T must both be cells'); end 

if nargin<3, options = struct(); end

N = length(data);

for n = 1:N
    if ischar(data{n})
        fsub = data{n};
        loadfile_sub;
    else
        X = data{n};
    end
    X = preprocdata(X,T{n},options);
    if n == 1
        ndim = size(X,2);
        V = zeros(ndim,ndim,N);
    end
    V(:,:,n) = X' * X;
end

D = NaN(N);
try
    for n1 = 1:N-1
        for n2 = n1+1:N
            D(n1,n2) =  ( wishart_kl(V(:,:,n1),V(:,:,n2),sum(T{n1}),sum(T{n2})) + ...
                wishart_kl(V(:,:,n2),V(:,:,n1),sum(T{n2}),sum(T{n1})) ) /2;
            D(n2,n1) = D(n1,n2);
        end
    end
catch
    for n = 1:N
        V(:,:,n) = V(:,:,n) + 0.0001*eye(ndim);
    end
    for n1 = 1:N-1
        for n2 = n1+1:N
            D(n1,n2) =  ( wishart_kl(V(:,:,n1),V(:,:,n2),sum(T{n1}),sum(T{n2})) + ...
                wishart_kl(V(:,:,n2),V(:,:,n1),sum(T{n2}),sum(T{n1})) ) /2;
            D(n2,n1) = D(n1,n2);
        end
    end    
end

end


function X = preprocdata(X,T,options)

% Filtering
if isfield(options,'filter') && ~isempty(options.filter)
    X = filterX(X,T,options.Fs,options.filter);
end
% Detrend X
if isfield(options,'detrend') && options.detrend
    X = detrendX(X,T);
end
% Hilbert envelope
if isfield(options,'onpower') && options.onpower
    X = rawsignal2power(X,T);
end
% Embedding
if isfield(options,'embeddedlags') && length(options.embeddedlags) > 1
    X = embeddata(X,T,options.embeddedlags);
end
X = zscore(X);

end