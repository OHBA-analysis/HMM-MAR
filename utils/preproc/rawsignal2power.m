function data = rawsignal2power(data,T)
% Gets *square of* the power time series using the Hilbert transform,
% as done in Baker et al (2014), eLife.
%
% Diego Vidaurre, OHBA, University of Oxford (2017)

if isstruct(data), ndim = size(data.X,2);
else, ndim = size(data,2);
end

for n = 1:length(T)
    ind = sum(T(1:n-1))+1:sum(T(1:n));
    if isstruct(data)
        for j = 1:ndim
            data.X(ind,j) = abs(hilbert(data.X(ind,j)));
        end
    else
        for j = 1:ndim
            data(ind,j) = abs(hilbert(data(ind,j)));
        end
    end
end

end
