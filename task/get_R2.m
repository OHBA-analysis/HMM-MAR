function R2 = get_R2(Y,Ypred,lossfunc)
d = Y - Ypred;
if strcmp(lossfunc,'quadratic')
    l = d.^2; l0 = Y.^2; ee = 1/2;
elseif strcmp(lossfunc,'absolute')
    l = abs(d); l0 = abs(Y); ee = 1;
elseif strcmp(lossfunc,'huber')
    l = zeros(size(d)); l0 = zeros(size(d)); ee = 1;
    for j1 = 1:N
        for j2 = 1:q
            ii = abs(d(:,j1,j2))<1; l(ii,j1,j2) = d(ii,j1,j2).^2;
            ii = abs(d(:,j1,j2))>=1; l(ii,j1,j2) = abs(d(ii,j1,j2));
            ii = abs(Y(:,j1,j2))<1; l0(ii,j1,j2) = Y(ii,j1,j2).^2;
            ii = abs(Y(:,j1,j2))>=1; l0(ii,j1,j2) = abs(Y(ii,j1,j2));
        end
    end
end
% across-trial R2, using mean of euclidean distances in stimulus space
if length(l)==size(l,1)
    m = mean(sum(l).^(ee),2);
    m0 = mean(sum(l0).^(ee),2);
    R2 = 1 - m ./ m0;
else
    m = mean(sum(l,3).^(ee),2);
    m0 = mean(sum(l0,3).^(ee),2);
    R2 = 1 - m ./ m0;
end
% % mean of R2, one per trial - equivalent to the previous
% m = sum(l,3).^(ee);
% m0 = sum(l0,3).^(ee);
% R2 = mean(1 - m ./ m0,2);
% % computing SE with all trials and features at once
% se = sum(sum(l,3),2).^(ee);
% se0 = sum(sum(l0,3),2).^(ee);
% R2 = 1 - se ./ se0;
end
