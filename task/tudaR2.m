function [R2,R2_states] = tudaR2(X,Y,T,tuda,Gamma)
% Training explained variance per time point per each state
%
% R2_states is per state, R2 is for the entire model
% X and Y must be in matrix format (2D)
% All trials must have equal size, ie all elements of T must be equal
%
% Diego Vidaurre

if any(T(1)~=T)
    error(['tudaR2 requires that all trials have the same length, ' ...
        'i.e. that all elements in T are equal']);
end

[X,Y,T] = preproc4hmm(X,Y,T,tuda.train);

intercept = all(X(:,1)==1); 

N = length(T); ttrial = T(1); p = size(X,2);
K = length(tuda.state); q = size(Y,2);

if intercept
    mY = repmat(mean(Y),size(Y,1),1); % one single mean value for the entire data
    mY = reshape(mY,[ttrial N q]);
else
    mY = zeros(ttrial,N,q);
end

mat1 = ones(ttrial,q);
Y = reshape(Y,[ttrial N q]);
Yhat = zeros(ttrial,N,q);

R2_states = zeros(ttrial,q,K);
beta = tudabeta(tuda);
for k = 1:K
    Yhatk = X * beta(:,:,k); %tuda.state(k).W.Mu_W(1:p,p+1:end);
    Yhatk = reshape(Yhatk,[ttrial N q]);
    if k > size(Gamma,2) % NESS model baseline state
        break%
        noGamma = prod(1-Gamma,2);
        Yhat = Yhat + Yhatk .* reshape(repmat(noGamma,1,q),[ttrial,N,q]);
    else
        Yhat = Yhat + Yhatk .* reshape(repmat(Gamma(:,k),1,q),[ttrial,N,q]);
    end
    if intercept
        mYk = mean(reshape(Y,[ttrial*N q]) .* repmat(Gamma(:,k),1,q));
        mYk = reshape(repmat(mYk,size(Y,1),1),[ttrial N q]);
    else
        mYk = zeros(ttrial,N,q);
    end
    ek = permute(sum((Yhatk - Y).^2,2),[1 3 2]); % ttrial x q
    e0k = permute(sum((mYk - Y).^2,2),[1 3 2]); % ttrial x q
    R2_states(:,:,k) = mat1 - ek ./ e0k ; % we do not compute the   
end

e = permute(sum((Yhat - Y).^2,2),[1 3 2]); % ttrial x q
e0 = permute(sum((mY - Y).^2,2),[1 3 2]); % ttrial x q
R2 = mat1 - e ./ e0 ;

end