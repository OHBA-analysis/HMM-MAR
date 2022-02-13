function [Gamma,Xi,L] = fb_Gamma_inference_ehmm(XX,ehmm,residuals,Gamma,T)
% Inference using forward backward propagation for parallel ehmm chains

% Note that this is an approximation, so there can be small increments of
% the free energy in this step. This is because in the free energy we
% compute the weighted-average response and then get the squared error
% (and same thing in the update of the state distributions), whereas here
% we do the squaring per chain first. One solution would be to compute the
% free energy for each combination of states (2^K) and do likewise for the
% estimation of the states but this is too slow and possibly less accurate
% in terms of the state estimation. The other solution would be to devise
% some way to do the forward-backward equations all at once, combining
% likelihoods and then squaring, but that's not very easy to do.

order = ehmm.train.maxorder;
if nargin<5, T = size(residuals,1) + order; ttrial = T(1); N = 1;
else, ttrial = T(1); N = length(T);
end % normally this is called for a continuous segment
K = ehmm.K; show_warn = 1;

Xi = zeros(ttrial-1-order,K,4);

if ehmm.train.acrosstrial_constrained
    Gamma = squeeze(mean(reshape(Gamma,[ttrial-order,N,K]),2));
end
    

for k = 1:K % %randperm(K) % chain K+1 is implicit
    
    Lk = zeros(ttrial * N, 2); % N > 1 only if we are averaging across trials
    
    for j = 1:N % N is 1 except when acrosstrial_constrained
        ind = (1:ttrial) + ttrial * (j-1);
        ind2 = (1:(ttrial-order)) + (ttrial-order) * (j-1);
        Lk(ind,:) = obslike_ehmm(ehmm,Gamma,residuals(ind2,:),XX(ind2,:),k);
    end
    
    if ehmm.train.acrosstrial_constrained
       Lk = squeeze(sum(reshape(Lk,[ttrial,N,2]),2)); 
    end
    
    scale = zeros(ttrial,1);
    alpha = zeros(ttrial,2);
    beta = zeros(ttrial,2);
    
    alpha(1+order,:) = ehmm.state(k).Pi.*Lk(1+order,:);
    scale(1+order) = sum(alpha(1+order,:));
    alpha(1+order,:) = alpha(1+order,:)/scale(1+order);
    for i = 2+order:ttrial
        alpha(i,:) = (alpha(i-1,:)*ehmm.state(k).P).*Lk(i,:);
        scale(i) = sum(alpha(i,:));		% P(X_i | X_1 ... X_{i-1})
        alpha(i,:) = alpha(i,:)/scale(i);
    end
    
    scale(scale<realmin) = realmin;
    
    beta(ttrial,:) = ones(1,2)/scale(ttrial);
    for i = ttrial-1:-1:1+order
        beta(i,:) = (beta(i+1,:).*Lk(i+1,:))*(ehmm.state(k).P')/scale(i);
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
    Gamma(:,k) = Gamma2(order+1:ttrial,1);
    
    if out_precision
        xi = approximateXi(Gamma2(order+1:ttrial,:),size(Gamma2,1),ehmm);
        Xi(:,k,:) = reshape(xi,[size(xi,1) size(xi,2)*size(xi,3)]);
        if show_warn
            show_warn = false;
            warning('out of precision when computing xi')
        end
    else
        for i = 1+order:ttrial-1
            t = ehmm.state(k).P .* ( alpha(i,:)' * (beta(i+1,:).*Lk(i+1,:)));
            Xi(i-order,k,:) = t(:)'/sum(t(:));
        end
    end
    
end

L = scale; % only the last one

end