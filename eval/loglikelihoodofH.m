function [Q_h] = loglikelihoodofH(Y,X,hmm,t)
% function calculates the log likelihood of our approximate likelihood
% function h(Y|w,gamma,xi), which approximates the likelihood P(Y|w,gamma)
% for bayesian weighted logistic regression models.

[T,~]=size(Y);
[T2,dimX]=size(X);
n=dimX+1;
if T>1 || T2>1 
    ME=MException(loglikelihoodofH:tunidimensional, ...
        'Error: this function only takes single point values in time; received vector across time');
    throw ME;
end
% if dimX ~= size(hmm.state(1).w_mu,1)
%     ME=MException(loglikelihoodofH:xdimension, ...
%         'Error: X dimension inputted does not match W dimension');
%     throw ME;
% end
if (sum(Y==-1)+sum(Y==1))~=T
    MException(loglikelihoodofH:Yformaterror,'Error; Y not in correct format for binary inference')
    throw ME;
end

if ~isfield(hmm,'psi')
    MException(loglikelihoodofH:nopsirec,'Error; No record of PSI in hmm structure')
    throw ME;
end

Sind=hmm.train.S==1;


% calculation broken down into component parts for ease of reference:
comp1 = log (logsig(hmm.psi(t)));

comp2 = 0;
for i=1:length(hmm.state)
    comp2= comp2 + hmm.Gamma(t,i) * X * hmm.state(i).W.Mu_W(Sind);
end
comp2 = (Y*comp2 - hmm.psi(t))/2;

lambdafunc = @(xi_in) ((2*xi_in).^-1).*(logsig(xi_in)-0.5);
comp3 = -(hmm.psi(t).^2);
for i=1:length(hmm.state)
    comp3 = comp3 + hmm.Gamma(t,i) * X * ...
        (squeeze(hmm.state(i).W.S_W(n,Sind(:,n),Sind(:,n)))+hmm.state(i).W.Mu_W(Sind)*hmm.state(i).W.Mu_W(Sind)') * X';
    %(params.state(i).w_mu*params.state(i).w_mu') * X';
end
comp3 = lambdafunc(hmm.psi(t))*comp3;

Q_h = comp1 + comp2 - comp3;

end