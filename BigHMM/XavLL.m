function avLL = XavLL(X,T,metastates,Gamma,options)
N = length(Gamma); K = length(metastates);
avLL = zeros(N,1); ndim = size(X,2); 
ltpi = ndim/2 * log(2*pi);
tacc = 0; 
orders = options.orders;
for i = 1:N
    T_i = T{i};
    X_i = X( (1:sum(T_i)) + tacc,:); tacc = tacc + sum(T_i);
    XX = formautoregr(X_i,T_i,orders,options.order,options.zeromean);
    X_i = X_i(1+options.order:end,:);
    Gamma_i = Gamma{i};
    for k=1:K
        m = metastates(k);
        NormWishtrace = zeros(sum(T_i),1);
        if isvector(m.Omega.Gam_rate)
            ldetWishB=0;
            PsiWish_alphasum=0;
            C = m.Omega.Gam_shape ./ m.Omega.Gam_rate;
            for n=1:ndim,
                ldetWishB=ldetWishB+0.5*log(m.Omega.Gam_rate(n));
                PsiWish_alphasum=PsiWish_alphasum+0.5*psi(m.Omega.Gam_shape);
                if ndim==1
                    NormWishtrace =  0.5 * C(n) * sum( (XX * m.W.S_W) .* XX, 2);
                else
                    NormWishtrace = NormWishtrace + 0.5 * C(n) * ...
                            sum( (XX * permute(m.W.S_W(n,:,:),[2 3 1])) .* XX, 2);
                end
            end;
            avLL(i) = avLL(i) + sum(Gamma_i(:,k)) * (-ltpi-ldetWishB+PsiWish_alphasum);
        else 
            ldetWishB=0.5*logdet(m.Omega.Gam_rate);
            PsiWish_alphasum=0;
            C = m.Omega.Gam_shape * m.Omega.Gam_irate;
            if isempty(orders)
                NormWishtrace = 0.5 * sum(sum(C .* m.W.S_W));
            else
                I = (0:length(orders)*ndim+(~options.zeromean)-1) * ndim;
            end
            for n=1:ndim
                PsiWish_alphasum=PsiWish_alphasum+0.5*psi(m.Omega.Gam_shape/2+0.5-n/2);
                if ~isempty(orders)
                    index1 = I + n1;  
                    tmp = (XX * m.W.S_W(index1,:));
                    for n2=1:ndim
                        index2 = I + n2;  
                        NormWishtrace = NormWishtrace + 0.5 * C(n1,n2) * ...
                            sum( tmp(:,index2) .* XX,2);
                    end
                end
            end;
            avLL(i) = avLL(i) + sum(Gamma{i}(:,k)) * (-ltpi-ldetWishB+PsiWish_alphasum);
            NormWishtrace = 0.5 * sum(sum(m.W.S_W .* C));
        end
        d = X_i - XX * m.W.Mu_W;
        if isvector(m.Omega.Gam_rate)
            Cd =  repmat(C',1,sum(T{i})) .* d';
        else
            Cd = C * d';
        end
        dist=zeros(sum(T{i}),1);
        for n=1:ndim,
            dist=dist-0.5*d(:,n).*Cd(n,:)';
        end
        avLL(i) = avLL(i) + sum(Gamma{i}(:,k).*(dist - NormWishtrace));
    end
end
end
