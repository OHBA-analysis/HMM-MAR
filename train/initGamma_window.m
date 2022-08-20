function Gamma = initGamma_window(data,T,options)
% Window-based init
%
% Author: Diego Vidaurre, Aarhus University (2022)       

winsize = options.init_windowsize;
N = length(T); ndim = size(data.X,2); 

B = zeros(sum(T) - N*(winsize-1),options.order*ndim^2);
c = 1; 
for j = 1:N
    for t = 1:T(j)-winsize+1
        ind = sum(T(1:j-1)) + (t-1) + (1:winsize);
        b = mlmar(data.X(ind,:),options.order);
        B(c,:) = b(:); 
        c = c + 1; 
    end
end
% options = rmfield(options,'orders'); 
% options = rmfield(options,'S');
% options = rmfield(options,'Sind'); 
% options.ndim = size(B,2);
% options.order = 0; 
% options.zeromean = 0;
% options.covtype = 'shareddiag';
% options.initrep = 1; 
% options.inittype = 'random'; 
% options.cyc = 100; 
% [~,Gamma0] = hmmmar(B,T-winsize+1,options); 
Gamma0 = kmeans(B,options.K,'Replicates',options.initrep);
Gamma0 = vpath_to_stc(Gamma0,options.K);
Gamma = zeros(sum(T),options.K); I = true(sum(T),1); 
for j = 1:N
    ind = sum(T(1:j-1)) + (1:T(j));
    I(ind(1:options.order)) = false; 
    ind0 = sum(T(1:j-1)) - (j-1)*winsize + (1:T(j)-winsize+1);
    Gamma(ind(winsize/2:end-winsize/2),:) = Gamma0(ind0,:);
    Gamma(ind(1:winsize/2-1),:) = repmat(Gamma(ind(winsize/2),:),winsize/2-1,1);
    Gamma(ind(end-winsize/2+1:end),:) = repmat(Gamma(ind(end-winsize/2),:),winsize/2,1);
end
Gamma = Gamma(I,:); 

end

