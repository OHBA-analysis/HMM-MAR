function X = flipdata(X,T,flips)
% Flip the channels of X according to flips (no. trials x no.channels),
% which is the output of findflip.
%
% Author: Diego Vidaurre, University of Oxford.

N = length(T); ndim = size(X,2);
for in = 1:N
    ind = (1:T(in)) + sum(T(1:in-1));
    for d = 1:ndim
        if flips(in,d)
            X(ind,d) = -X(ind,d);
        end
    end
end
end