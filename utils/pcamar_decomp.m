function [B,E,C] = pcamar_decomp (X,T,options)

is_cell_strings = iscell(X) && ischar(X{1});
is_cell_matrices = iscell(X) && ~ischar(X{1});
is_struct = ~is_cell_strings && ~is_cell_matrices && isstruct(X);

orders = formorders(options.order,options.orderoffset,options.timelag,options.exptimelag);
 
if is_cell_strings || is_cell_matrices
    [~,XX,Y] = loadfile(X{1},T{1},options);
    XX2 = (XX' * XX); XY = (XX' * Y);
    for n=2:length(T)
        [~,XX,Y] = loadfile(X{n},T{n},options);
        XX2 = XX2 + (XX' * XX);
        XY = XY + (XX' * Y);
    end
elseif is_struct
    Y = zeros(sum(T)-length(T)*options.order,size(X.X,2));
    for n=1:length(T)
        t0 = sum(T(1:n-1));
        t = sum(T(1:n-1)) - (n-1)*options.order;
        Y(t+1:t+T(n)-options.order,:) = X.X(t0+options.order+1:t0+T(n),:);
    end
    XX = formautoregr(X.X,T,orders,options.maxorder,options.zeromean,1);
    XX2 = (XX' * XX); XY = (XX' * Y);
else
    Y = zeros(sum(T)-length(T)*options.order,size(X,2));
    for n=1:length(T)
        t0 = sum(T(1:n-1));
        t = sum(T(1:n-1)) - (n-1)*options.order;
        Y(t+1:t+T(n)-options.order,:) = X(t0+options.order+1:t0+T(n),:);
    end
    XX = formautoregr(X,T,orders,options.maxorder,options.zeromean,1);
    XX2 = (XX' * XX); XY = (XX' * Y);
end

ndim = size(Y,2);
W = (XX2 + 1e-9 * diag(size(XX,2))) \ XY; % (~zeromean)+ndim*length(orders) x ndim
W = W((1+(~options.zeromean)):end,:); % remove mean
Mu_W = zeros(ndim,length(orders),ndim);
for n=1:ndim
   Mu_W(:,:,n) = reshape(W(:,n),ndim,length(orders));
end
Mu_W = permute(Mu_W,[1 3 2]); % make it (ndim x ndim x orders)
B = zeros(ndim,options.pcamar,length(orders));
E = zeros(options.pcamar,options.pcamar,length(orders));
C = zeros(options.pcamar,ndim,length(orders));

for j=1:length(orders)
    [U,D,V] = svd(Mu_W(:,:,j));
    B(:,:,j) = U(:,1:options.pcamar);
    E(:,:,j) = D(1:options.pcamar,1:options.pcamar);
    C(:,:,j) = V(1:options.pcamar,:);
end


end
