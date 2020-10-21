 function [Gamma,Xi,L] = fb_Gamma_inference_addHMM(XX,hmm,residuals,Gamma)
% inference using forward backward propagation for the additive HMM

if isfield(hmm.train,'distribution') && strcmp(hmm.train.distribution,'logistic')
    order = 0;
else
    order = hmm.train.maxorder;
end
T = size(residuals,1) + order;
K = hmm.K;
update_residuals = hmm.train.order > 0 || hmm.train.zeromean == 0; 
additiveCovMat = (strcmpi(hmm.train.covtype,'full') || strcmpi(hmm.train.covtype,'diag'));
if additiveCovMat
    L = obslike([],hmm,residuals,XX,hmm.cache,[],Gamma);
else
    L = obslike([],hmm,residuals,XX,hmm.cache);
end
if ~update_residuals 
    Lk = L; % conditional to other chains
end
Xi = zeros(T-1-order,K,4);

if update_residuals 
    meand = computeStateResponses(XX,size(residuals,2),hmm,Gamma,1:K,true);
else
    residuals_k = residuals;
end

for k = 1:K %randperm(K) % chain K+1 is implicit
    
    if update_residuals
        meand_k = computeStateResponses(XX,size(residuals,2),hmm,Gamma,k,true);
        meand = meand - meand_k;
        residuals_k = residuals - meand;
    end
    % conditional to other chains
    if additiveCovMat
        Lk = obslike([],hmm,residuals_k,XX,hmm.cache,[k K+1],Gamma);
    else
        Lk = obslike([],hmm,residuals_k,XX,hmm.cache,[k K+1]);
    end
    
    scale = zeros(T,1);
    alpha = zeros(T,2);
    beta = zeros(T,2);
    
    alpha(1+order,:) = hmm.state(k).Pi.*Lk(1+order,:);
    scale(1+order) = sum(alpha(1+order,:));
    alpha(1+order,:) = alpha(1+order,:)/scale(1+order);
    for i = 2+order:T
        alpha(i,:) = (alpha(i-1,:)*hmm.state(k).P).*Lk(i,:);
        scale(i) = sum(alpha(i,:));		% P(X_i | X_1 ... X_{i-1})
        alpha(i,:) = alpha(i,:)/scale(i);
    end
    
    scale(scale<realmin) = realmin;
    
    beta(T,:) = ones(1,2)/scale(T);
    for i = T-1:-1:1+order
        beta(i,:) = (beta(i+1,:).*Lk(i+1,:))*(hmm.state(k).P')/scale(i);
        beta(i,beta(i,:)>realmax) = realmax;
    end
    
    if any(isnan(beta(:)))
        warning(['State time courses became NaN (out of precision). ' ...
            'There are probably extreme events in the data. ' ...
            'Using an approximation..' ])
        Gamma2 = alpha; out_precision = true;
    else
        Gamma2 = (alpha.*beta); out_precision = false; 
    end
    
    Gamma2 = rdiv(Gamma2,sum(Gamma2,2));
    Gamma(:,k) = Gamma2(:,1);
    
    if out_precision
        xi = approximateXi(Gamma,size(Gamma,1)+order,hmm);
        Xi(:,k,:) = reshape(xi,[size(xi,1) size(xi,2)*size(xi,3)]);
    else
        for i = 1+order:T-1
            t = hmm.state(k).P.*( alpha(i,:)' * (beta(i+1,:).*Lk(i+1,:)));
            Xi(i-order,k,:) = t(:)'/sum(t(:));
        end
    end
    
    if update_residuals
        meand_k = computeStateResponses(XX,size(residuals,2),hmm,Gamma,k,true);
        meand = meand + meand_k;
    end
    
end

Gamma = Gamma(1+order:T,:);

end