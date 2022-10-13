beta1 = hmm.state(k).W.Mu_W(Sind(:,n),n);

XXX = XX(Gamma(:,k) == 1,1:63); YYY = residuals(Gamma(:,k) == 1,end);
beta2 = (XXX' * XXX) \ (XXX' * YYY);


beta3 =  ((XX(:,1:63) .* Gamma(:,k))' * XX(:,1:63) ) \ ...
    ((XX(:,1:63) .* Gamma(:,k))' * residuals(:,end) );


scatter(beta1,beta2)
ylim([-.2 .2]);xlim([-.2 .2])


XXGXX2 = (XXX' * XXX); 
XXGXX3 = ((XX(:,1:63) .* Gamma(:,k))' * XX(:,1:63) ); 
figure(2); subplot(121);imagesc(XXGXX2); subplot(122);imagesc(XXGXX3-XXGXX2);colorbar

XXX = XX(Gamma(:,k) == 1,1:63); YYY = residuals(Gamma(:,k) == 1,end);
XY2 = (XXX' * YYY);
XY3 = ((XX(:,1:63) .* Gamma(:,k))' * residuals(:,end) );
figure(3); subplot(121);imagesc(XY2); colorbar;subplot(122);imagesc(XY3);colorbar



XY4 = ((XX(:,1:63) .* Gamma(:,k))' * (residuals(:,end) .* Gamma(:,k)  ) );
XXX2 = XX(:,1:63) .* Gamma(:,k);

YYY = residuals(Gamma(:,k) == 1,end);
YYY2 = residuals(:,end) .* Gamma(:,k) ;
[sum(YYY) sum(YYY2)]


regterm2 = zeros(size(regterm));


hmm.state(k).W.iS_W(n,Sind(:,n),Sind(:,n)) = ...
    regterm + Tfactor * c * XXGXX{k}(Sind(:,n),Sind(:,n));
hmm.state(k).W.iS_W(n,Sind(:,n),Sind(:,n)) = ...
    (permute(hmm.state(k).W.iS_W(n,Sind(:,n),Sind(:,n)),[2 3 1]) + ...
    permute(hmm.state(k).W.iS_W(n,Sind(:,n),Sind(:,n)),[2 3 1])' ) / 2; % ensuring symmetry
hmm.state(k).W.S_W(n,Sind(:,n),Sind(:,n)) = ...
    inv(permute(hmm.state(k).W.iS_W(n,Sind(:,n),Sind(:,n)),[2 3 1]));
sx = permute(hmm.state(k).W.S_W(n,Sind(:,n),Sind(:,n)),[2 3 1]) * ...
    Tfactor * c * XX(:,Sind(:,n))';
hmm.state(k).W.Mu_W(Sind(:,n),n) = (sx .* Gamma(:,k)') * residuals(:,n);

beta12 = hmm.state(k).W.Mu_W(Sind(:,n),n);


scatter(beta12,beta2)
ylim([-.2 .2]);xlim([-.2 .2])  


%%

figure(6)

b = squeeze(Betas(:,:,1,icv));
Xtrain1 = reshape(X(1:3,c.training{icv},:),[Ntr*3 p] ) ;
Ytrain1 = reshape(Y(1:3,c.training{icv},:),[Ntr*3 1] ) ;
b2 = pinv(Xtrain1) * Ytrain1;

scatter(b,b2)
ylim([-.2 .2]);xlim([-.2 .2])  
