function V = pcapred_decomp (X,T,options)
% it assumes that X is centered

is_cell_strings = iscell(X) && ischar(X{1});
is_cell_matrices = iscell(X) && ~ischar(X{1});
is_struct = ~is_cell_strings && ~is_cell_matrices && isstruct(X);

orders = formorders(options.order,options.orderoffset,options.timelag,options.exptimelag);

N = 0; 
if is_cell_strings || is_cell_matrices
    [~,XX] = loadfile(X{1},T{1},options);
    if options.zeromean==0, XX = XX(:,2:end); end
    XX2 = (XX' * XX);  
    N = N + size(XX,1);
    for n=2:length(T)
        [~,XX] = loadfile(X{n},T{n},options);
        if options.zeromean==0, XX = XX(:,2:end); end
        XX2 = XX2 + (XX' * XX);
        N = N + size(XX,1);
    end
elseif is_struct
    XX = formautoregr(X.X,T,orders,options.maxorder,1,1);
    N = size(XX,1);
    XX2 = (XX' * XX); 
else
    XX = formautoregr(X,T,orders,options.maxorder,1,1);
    N = size(XX,1);
    XX2 = (XX' * XX);  
end

[V,~,~] = svd( XX2 / (N-1) );
V = V(:,(1:options.pcapred) + (options.vcomp-1) );

end
