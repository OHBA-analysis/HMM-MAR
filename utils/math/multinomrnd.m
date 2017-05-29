function [y,p_avg,p_std]=multinomrnd(p,m,n)
%Performs random sampling from a binomial distribution
%
% [y]=multinomrnd(p,m,n)
% where p=1-by-k vector of probabilities of occurrence 
%       n=sample size
% and   m= number of trials
%       y=samples-matrix of size k-by-m
%
% for picking out one of k mixture components, set n=1;
%
if nargin<3
  n=1;
end

k=length(p);
x=rand(n,m);

if (sum(p)-1>100*eps) 
  p(k+1)=1-sum(p); 
  k=k+1; 
end
p=cumsum(p);


y(1,:)=sum(x<=p(1),1);
for i=2:k
  y(i,:)=sum(x>p(i-1) & x<=p(i),1);
end

p_avg=mean(y'./n);
p_std=std(y'./n);