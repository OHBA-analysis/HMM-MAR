function [Y_orth,Y_partialled] = decorrelateDesignMatrix(Y)
% Given a design matrix Y, this method decorrelates regressors using
% symmetric orthogonalisation. This approach finds the timecourse Y_orth 
% with zero correlation between columns that minimises (Y-Y_orth)^2.
% This approach is best used when wanting to interpret the effects of each
% variable in the design matrix whilst removing common effects.
%
% This method also returns a matrix Y_partialled containing the residual of
% each column after regressing out any effects of all remaining columns.
% This method should be used when wishing to focus on one variable in the
% design matrix whilst controlling for all other (nuisance) variates; note
% that the columns of Y_partialled are NOT orthogonal to each other,
% but they are orthogonal with respect to the original columns of Y and 
% should be interpreted appropriately.
Y = normalise(Y,1);
temp = ROInets.remove_source_leakage(Y', 'symmetric');
Y_orth = temp';

if nargout>1
    [T,N] = size(Y);
    for i=1:N
        [~,~,Y_partialled(:,i)] = regress(Y(:,i),Y(:,setdiff(1:N,i)));
    end
end
end