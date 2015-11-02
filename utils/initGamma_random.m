function Gamma = initGamma_random(T,K,D)
Gamma = zeros(sum(T),K);
P = ones(K,K); 
P = P - diag(diag(P)) + D*eye(K);
for k=1:K
    P(k,:)=P(k,:)./sum(P(k,:),2);
end
for tr=1:length(T)
    gamma = zeros(T(tr),K);
    gamma(1,:) = mnrnd(1,ones(1,K)*1/K);
    for t=2:T(tr)
        gamma(t,:) = mnrnd(1,P(find(gamma(t-1,:)==1),:));
    end
    gamma = gamma + 0.0001 * rand(T(tr),K);
    t0 = sum(T(1:tr-1)); t1 = sum(T(1:tr));
    Gamma(t0+1:t1,:) = gamma ./ repmat(sum(gamma,2),1,K);
end
end