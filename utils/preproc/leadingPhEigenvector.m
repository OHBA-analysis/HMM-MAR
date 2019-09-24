function data = leadingPhEigenvector(data,T)
% Computes the leading eigenvectors of the time series of phase coherence.
% That is,
% - Calculates the iFC (or instantaneous coherence matrix)
% - Calculates the instantaneous Leading Eigenvector
% - Calculates the instantaneous % of variance
% - Calculates de time x time FCD matrices
% This procedure is decribed in detail in
% Cabral et al. (2017), Scientific Reports.
%
% Diego Vidaurre, OHBA, University of Oxford (2017)
% Joana Cabral, University of Oxford (2017)

if isstruct(data), ndim = size(data.X,2);
else, ndim = size(data,2);
end

for n = 1:length(T)
    ind = sum(T(1:n-1))+1:sum(T(1:n));
    Ph = zeros(length(ind),ndim);
    if isstruct(data)
        for j=1:ndim
            Ph(:,j) = angle(hilbert(data.X(ind,j)));
        end
    else
        for j=1:ndim
            Ph(:,j) = angle(hilbert(data(ind,j)));
        end
    end
    for t = 1:length(ind)
        iFC = ones(ndim);
        for j1 = 1:ndim-1
            for j2 = j1+1:ndim
                iFC(j1,j2) = cos(adif(Ph(t,j1),Ph(t,j2)));
                iFC(j2,j1) = iFC(j1,j2);
            end
        end
        [v,d] = eig(iFC); 
        [~,i] = max(diag(d));
        if isstruct(data)
            data.X(ind(t),:) = v(:,i);
        else
            data(ind(t),:) = v(:,i);
        end
    end
end
end


function c = adif(a,b)
d = abs(a-b);
if d > pi
    c = 2*pi-d;
else
    c = d;
end
end
