function [Gamma,Xi,scale] = fb_Gamma_inference_sub(L,P,Pi,T,order,constraint)

if nargin<4, T = size(L,1); end
if nargin<5, order = 0; end
if nargin<6, constraint = []; end
K = size(P,1);

L(L<realmin) = realmin;
L(L>realmax) = realmax;

scale = zeros(T,1);
alpha = zeros(T,K);
beta = zeros(T,K);

alpha(1+order,:) = Pi.*L(1+order,:);
scale(1+order) = sum(alpha(1+order,:));
alpha(1+order,:) = alpha(1+order,:)/scale(1+order);
for i = 2+order:T
    alpha(i,:) = (alpha(i-1,:)*P).*L(i,:);
    scale(i) = sum(alpha(i,:));		% P(X_i | X_1 ... X_{i-1})
    alpha(i,:) = alpha(i,:)/scale(i);
end

scale(scale<realmin) = realmin;

beta(T,:) = ones(1,K)/scale(T);
for i = T-1:-1:1+order
    beta(i,:) = (beta(i+1,:).*L(i+1,:))*(P')/scale(i);
    beta(i,beta(i,:)>realmax) = realmax;
end

if any(isnan(beta(:)))
    warning(['State time courses became NaN (out of precision). ' ...
        'There are probably extreme events in the data. ' ...
        'Using an approximation..' ])
    Gamma = alpha; out_precision = true; 
else
    Gamma = (alpha.*beta); out_precision = false; 
end

if ~isempty(constraint)
    try
        Gamma(1+order:T,:) = Gamma(1+order:T,:) .* constraint;
    catch
        error(['options.Gamma_constraint must be (trial time X K), ' ...
            ' and all trials must have the same length' ])
    end
end

Gamma = Gamma(1+order:T,:);
Gamma = rdiv(Gamma,sum(Gamma,2));

if out_precision
    Xi = approximateXi(Gamma,size(Gamma,1)+order,order);
    Xi = reshape(Xi,[size(Xi,1) size(Xi,2)*size(Xi,3)]);
else
    Xi = zeros(T-1-order,K*K);
    for i = 1+order:T-1
        t = P.*( alpha(i,:)' * (beta(i+1,:).*L(i+1,:)));
        Xi(i-order,:) = t(:)'/sum(t(:));
    end
end

scale = scale(1+order:end);

end