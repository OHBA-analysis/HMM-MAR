function [acc,acc_star,Ypred,Ypred_star,Gammapred,acc_Gamma,AUC,expLL] = tucacv(X,Y,T,options)
% Wrapper function to perform temporally unconstrained classification
% Analysis (TUCA). This function is now merely a wrapper function with core
% functionality maintained in tudacv.
if ~isfield(options,'classifier')
    options.classifier='logistic';
end

[acc,acc_star,Ypred,Ypred_star,Gammapred,acc_Gamma] = tudacv(X,Y,T,options);

if nargout>6
    q = size(Y,2);
    Ycopy = reshape(Y,[T(1),length(T),q]);
    for iF=1:size(acc_star,3)
        LL = Ycopy.*log(Ypred(:,:,:,iF));
        expLL(:,:,iF) = mean(sum(LL,3),2);

        Ypreds = Ypred_star(:,:,:,iF);
        for t=1:T(1)
            AUC_t = zeros(q);
            for i=1:q
                for j=(i+1):q
                    % find valid samples:
                    validtrials = union(find(Ycopy(t,:,i)),find(Ycopy(t,:,j)));
                    ytemp = permute(Ycopy(t,validtrials,[i,j]),[2,3,1]);
                    temp = exp(squeeze(Ypreds(t,validtrials,[i,j])) - max(squeeze(Ypreds(t,validtrials,[i,j])),[],2));
                    temp = rdiv(temp,sum(temp,2));
                    [temp,inds] = sort(temp(:,1),'descend');
                    ytemp = ytemp(inds,:);
                    for n=1:length(temp)
                        TPr(n) = sum(ytemp(1:n,1))/sum(ytemp(:,1));
                        p = temp(n,1);
                        FPr(n) = sum(ytemp(1:n,2))/sum(ytemp(:,2));
                    end
                    AUC_t(i,j) = sum(diff([0,FPr,1,1]) .* [0,TPr,1]);
                end
            end
            AUC(t,iF) = mean(AUC_t(logical(triu(ones(q),1))));
        end
    end
end
                    
end