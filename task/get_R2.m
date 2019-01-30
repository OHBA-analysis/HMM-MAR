function [R2,m] = get_R2(Y,Ypred,lossfunc,mode)
% Gets explained variance according to the loss function lossfunc
% If mode==1, then the first dimension of Y is time and the second number
%   of dimensions
% If mode==2, then the first dimension of Y is time, the second number
%   of trials, and the third the number of dimensions
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
if mode==1
    l = l(:); l0 = l0(:);
    m = sum(l).^(ee);
    m0 = sum(l0).^(ee);
    %m = mean(sum(l).^(ee),2);
    %m0 = mean(sum(l0).^(ee),2);
    R2 = 1 - m ./ m0;
else
    m = mean(sum(l,3).^(ee),2);
    m0 = mean(sum(l0,3).^(ee),2);
    R2 = 1 - m ./ m0;
end

end
