function X = sim_mlmar(w,Tnew,Xinit) 

% Simulate data from one single MAR trained with mlmar (only for MAR
% default options)
%
% Author: Diego Vidaurre, OHBA, University of Oxford (2021)

Nnew = length(Tnew);
ndim = size(w,2);
order = size(w,1) / ndim;
orders = 1:order;
if nargin<3
    d = 500;
else
    d = size(Xinit,1);
end

for n = 1:Nnew
    if nargin==3
        Xin = zeros(d+Tnew(n),ndim);
        Xin(1:d,:) = Xinit; 
        Xin(d+1:end,:) = randn(Tnew(n),ndim) .* repmat(std(Xinit),Tnew(n),1);
    else
        Xin = randn(d+Tnew(n),ndim);
    end
    
    for t = order+1:Tnew(n)+d
        XX = ones(1,order*ndim);
        for i = 1:length(orders)
            o = orders(i);
            XX(1,(1:ndim) + (i-1)*ndim) = Xin(t-o,:);
        end
        Xin(t,:) = Xin(t,:) + XX * w;
    end
    ind = (1:Tnew(n)) + sum(Tnew(1:n-1));
    X(ind,:) = Xin(d+1:end,:);
end


end
