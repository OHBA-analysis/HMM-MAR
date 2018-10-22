function [pval,d,side] = permtestdiff_aux(Xin,Yin,Nperm,confounds,side,func)
% tests if @func (mean by default) of Xin is higher than the mean of Yin 
% Diego Vidaurre, University of Oxford (2015)
% pval : pvalue
% d: the permutation values
% side: 1 if mean(Xin)>mean(Yin), -1 otherwise
% Diego Vidaurre

Xin = Xin(~isnan(Xin)); 
Yin = Yin(~isnan(Yin)); 

N1 = length(Xin); 
N2 = length(Yin); 
N = N1 + N2; 

if (nargin>3) && ~isempty(confounds)
    %if size(confoundsX,2)~=size(confoundsY,2)
    %    error('The confound features size should be the same for X and Y')
    %end
    if ~isempty(confounds)
        confounds = confounds - repmat(mean(confounds),N1,1);
        pconfounds = pinv(confounds);
        Xin = Xin - confounds * pconfounds * Xin;
        Yin = Yin - confounds * pconfounds * Yin;
    end
end

if nargin<6
    func = @mean;
end

d = zeros(Nperm+1,1); 
d(1) = func(Xin) - func(Yin);
if nargin<5
    %warning('Specification of side is required if you are going to do FDR later')
    if d(1)>0
        side = 1;
    else
        side = -1;
        d(1) = -d(1);
    end
else
    if side==-1
        d(1) = -d(1);
    end
end

if d(1)<0, pval = 1; return; end
Zin = [Xin; Yin];

for perm = 1:Nperm
   ind1 = randperm(N,N1);
   ind2 = setdiff(1:N,ind1);
   Xin = Zin(ind1);
   Yin = Zin(ind2);
   if side==1
       d(perm+1) = func(Xin) - func(Yin);
   else
       d(perm+1) = func(Yin) - func(Xin);
   end
       
end

pval = sum(d >= d(1)) / (Nperm+1); 

end