function GammaInit = initGamma_seq(T,K)

GammaInit = zeros(sum(T),K);
for j = 1:length(T)
    ind = (1:T(j)) + sum(T(1:j-1));
    assig = ceil(K*(1:T(j))./T(j));
    Gammaj = zeros(T(j), K);
    for k = 1:K
        Gammaj(assig==k,k) = 1;
    end
    GammaInit(ind,:) = Gammaj;
end

end
