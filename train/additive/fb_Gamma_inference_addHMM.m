function [Gamma,Xi,L] = fb_Gamma_inference_addHMM(XX,hmm,residuals,Gamma)
% Inference using forward backward propagation for the additive HMM

% Note that this is an approximation, so there can be small increments of
% the free energy in this step. This is because in the free energy we
% compute the weighted-average response and then get the squared error
% (and same thing in the update of the state distributions), whereas here 
% we do the squaring per state first. One solution would be to compute the
% free energy for each combination of states (2^K) and do likewise for the
% estimation of the states but this is too slow and possibly less accurate
% in terms of the state estimation. The other solution would be to devise
% some way to do the forward-backward equations all at once, combining
% likelihoods and then squaring, but that's not very easy to do. 

order = hmm.train.maxorder;
T = size(residuals,1) + order;
K = hmm.K;

Xi = zeros(T-1-order,K,4);
meand = computeStateResponses(XX,hmm,Gamma);

show_warn = true;
for k = 1:K %randperm(K) % chain K+1 is implicit
    
    meand_k = computeStateResponses(XX,hmm,Gamma,k,true);
    meand = meand - meand_k;
    residuals_k = residuals - meand;
    
    % conditional to other chains
    Lk = obslike_addHMM(hmm,residuals_k,XX,[k K+1]);
        
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
        if show_warn
            show_warn = false;
            warning('out of precision when computing xi')
        end
    else
        for i = 1+order:T-1
            t = hmm.state(k).P.*( alpha(i,:)' * (beta(i+1,:).*Lk(i+1,:)));
            Xi(i-order,k,:) = t(:)'/sum(t(:));
        end
    end

    meand_k = computeStateResponses(XX,hmm,Gamma,k,true);
    meand = meand + meand_k;
    
end

Gamma = Gamma(1+order:T,:);

if nargout > 2
    L = obslike_addHMM(hmm,residuals,XX,[]);
end

end