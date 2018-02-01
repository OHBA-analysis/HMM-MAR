function [hmm,fe] = dropstate(hmm,k,X,T)

K = length(hmm.state);
% if isfield(hmm.train,'grouping')
%     Q = length(unique(hmm.train.grouping));
% else
%     Q = 1;
% end
Q = 1; 

no_k = setdiff(1:K,k);
K = K - 1;
hmm.state(k) = [];
hmm.K = K; 
hmm.prior.Dir2d_alpha = hmm.prior.Dir2d_alpha(no_k,no_k);
hmm.prior.Dir_alpha = hmm.prior.Dir_alpha(no_k);
if Q==1
    hmm.Dir2d_alpha = hmm.Dir2d_alpha(no_k,no_k);
    hmm.Dir_alpha = hmm.Dir_alpha(no_k);
    hmm.P = hmm.P(no_k,no_k);
    hmm.Pi = hmm.Pi(no_k);
else
    hmm.Dir2d_alpha = hmm.Dir2d_alpha(no_k,no_k,:);
    hmm.Dir_alpha = hmm.Dir_alpha(no_k,:);
    hmm.P = hmm.P(no_k,no_k,:);
    hmm.Pi = hmm.Pi(no_k,:);
end

if nargout>1
   if iscell(T)
       fe = 0;
       for i=1:length(T)
           XX_i = cell(1);
           [X_i,XX_i{1},Y_i]  = loadfile(X{i},T{i},hmm.train);
           data = struct('X',X_i,'C',NaN(sum(T{i})-length(T{i})*hmm.train.order,K));
           [Gamma,~,Xi] = hsinference(data,T{i},hmm,Y_i,[],XX_i);
           fe = fe + sum(evalfreeenergy(X_i,T{i},Gamma,Xi,hmm,Y_i,XX_i,[1 1 1 0 0]));
       end
       fe = fe + sum(evalfreeenergy([],[],[],[],hmm,[],[],[0 0 0 1 1]));
   else
       [Gamma,~,Xi] = hsinference(X,T,hmm);
       fe = sum(evalfreeenergy(X,T,Gamma,Xi,hmm));
   end
end

end