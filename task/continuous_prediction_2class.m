function Ypred = continuous_prediction_2class(Y,Ypred)
% Which one is the most likely class, out of a continuous estimation?
[N,q] = size(Y); % N is time points
L = length(unique(Y(:))); 
if q == 1 && L == 2 % class is -1 or +1
    Ypred(Ypred<=0) = -1;
    Ypred(Ypred>0) = +1;
elseif q > 1 && L == 2 % dummy variable representation for classification
    [~,m] = max(Ypred,[],2);
    Ypred = zeros(size(Ypred));
    ind = sub2ind(size(Ypred),1:N, m');
    Ypred(ind) = 1; 
end
end