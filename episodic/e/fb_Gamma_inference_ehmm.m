function [Gamma,Xi,L] = fb_Gamma_inference_ehmm(XX,ehmm,residuals,Gamma,Xi,T)
% Inference using forward backward propagation for parallel ehmm chains

% Note that this is an approximation, so there can be small increments of
% the free energy in this step. This is because of two reasons:
% 1. The likelihood for the free energy and for the FB equations is
% computed slightly differently: the term NormWishtrace (related to the
% W covariance) is computed across states for the FB equations, and per
% state in the free energy. That's due to the fact that the FB equations
% are approximate here. 
% 2. The weights of how each chain contributes to each time point are *not*
% a linear combination of the Gamma state probabilities, because the way the weight 
% for the baseline state is computed is nonlinear (and therefore everything
% is not linear). In fact, the likelihood is also a nonlinear function of
% the state probabilities. For example, we could have the highest
% likelihood for gamma_kt = 0.3, but the forward-backward equations assume
% a linear progression between 0 and 1. A potential solution would be to
% bin the Gammas, and have L states per chain: 0% active, 20% active, etc.,
% but that complicates the definition of the trans prob matrices and their
% priors. 
%
% Author: Diego Vidaurre, University of Oxford / Aarhus University (2022)


order = ehmm.train.maxorder; K = ehmm.K; show_warn = true;

if nargin<6, T = size(residuals,1) + order; ttrial = T(1); N = 1;
else, ttrial = T(1); N = length(T);
end % normally this is called for a continuous segment
% if nargin<5 || isempty(Xi), Xi = zeros(ttrial-1-order,K,4); check_improv = false;
% else, check_improv = true;
% end
check_improv = false; 

if ehmm.train.acrosstrial_constrained
    Gamma = squeeze(mean(reshape(Gamma,[ttrial-order,N,K]),2));
end

if check_improv
    n_rejected = 0;
    if strcmpi(ehmm.train.stopcriterion,'LogLik')
        c = sum(evalfreeenergy_ehmm(T,Gamma,Xi,ehmm,residuals,XX,[0 1 1 0 0]));
    else
        c = sum(evalfreeenergy_ehmm(T,Gamma,Xi,ehmm,residuals,XX,[1 1 1 0 0]));
    end
    %disp(num2str(loglik))
end    

if isempty(Xi), Xi = zeros(ttrial-order-1,K,4); end

for k = randperm(K) 
 
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
        Gammak = alpha; out_precision = true;
    else
        Gammak = (alpha.*beta); out_precision = false;
    end
    
    Gammak = rdiv(Gammak,sum(Gammak,2));
    if check_improv, Gammak0 = Gamma(:,k); Xi0k = Xi(:,k,:); end
    Gamma(:,k) = Gammak(order+1:ttrial,1);
    
    if isempty(Xi), Xi = zeros(ttrial-order-1,K,4); end
    if out_precision
        xi = approximateXi(Gammak(order+1:ttrial,:),size(Gammak,1),ehmm);
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
    
    if check_improv
        if strcmpi(ehmm.train.stopcriterion,'LogLik')
            c2 = sum(evalfreeenergy_ehmm(T,Gamma,reshape(Xi,T-order-1,K,2,2),...
                ehmm,residuals,XX,[0 1 1 0 0]));
        else
            c2 = sum(evalfreeenergy_ehmm(T,Gamma,reshape(Xi,T-order-1,K,2,2),...
                ehmm,residuals,XX,[1 1 1 0 0]));
        end
        %disp(num2str(loglik2))
        if c2 > c % decrement of fitness, undo
            Gamma(:,k) = Gammak0; Xi(:,k,:) = Xi0k;
            n_rejected = n_rejected + 1;
        else
            c = c2;
        end
    end
    
end

if check_improv, disp(['%Change rejected: ' num2str(n_rejected/K) ]); end

L = scale; % only the last one

end