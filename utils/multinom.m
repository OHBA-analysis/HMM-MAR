function [y]=multinom(p,m,n)
%Performs random sampling from a binomial distribution
%
% [y]=multinom(p,m,n)
% where p=1-by-k vector of probabilities of occurrence 
%       n=sample size
% and   m= number of trials
%       y=samples-matrix of size k-by-m
%
% for picking out one of k mixture components, set n=1;
%
k=length(p);
x=rand(n,m);

if (sum(p)~=1) , 
  p(k+1)=1-sum(p); 
  k=k+1; 
end;
p=cumsum(p);

y(1,:)=sum(x<=p(1));
for i=2:k,
  y(i,:)=sum(x>p(i-1) & x<=p(i));
end;