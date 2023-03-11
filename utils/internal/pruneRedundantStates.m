function [hmm,Gamma,Xi,re_do_M] = pruneRedundantStates(hmm,Gamma,Xi,threshold)

if nargin < 4
    threshold = 0.05;
end

K = size(Gamma,2);

D = ones(K);
for k1 = 1:K-1
    for k2 = k1+1:K
        D(k1,k2) = mean(abs(Gamma(:,k1)-Gamma(:,k2))); % / sum(Gamma0(:,k1)) / sum(Gamma0(:,k2)) ;
    end
end

re_do_M = 0;
while any(D(:)<threshold)
    [~,ind] = min(D(:));
    [k1,k2] = ind2sub([K,K],ind);
    Gamma(:,k1) = Gamma(:,k1) + Gamma(:,k2);
    Gamma(:,k2) = 0;
    Xi(:,k1,:) = Xi(:,k1,:) + Xi(:,k2,:);
    Xi(:,k2,:) = 0; 
    Xi(:,:,k1) = Xi(:,:,k1) + Xi(:,:,k2);
    Xi(:,:,k2) = 0;  
    hmm.train.active(k2) = 0;
    D(k1,k2) = 1;
    re_do_M = 1;
end

end


