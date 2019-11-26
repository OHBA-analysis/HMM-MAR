function [acc,acc_star,Ypred,Ypred_star] = tucacv(X,Y,T,options)
% Wrapper function to perform temporally unconstrained classification
% Analysis (TUCA). This function is now merely a wrapper function with core
% functionality maintained in tudacv.
if ~isfield(options,'classifier')
    options.classifier='logistic';
end

[acc,acc_star,Ypred,Ypred_star] = tudacv(X,Y,T,options);

end