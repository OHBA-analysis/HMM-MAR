function pvalues = hmmtest_epoched(Gamma,T,Y,Nperm)
% Statistical testing on how the probability of the states is modulated
% by a stimulus, time point by time point. 
% For statistical testing on differences between groups of
% the average time spent on the different states, 
% or how quickly state switching happens, see hmmtest
%
% INPUTS
%
% Gamma: state time courses, as returned by hmmmar (time by states); 
%        or epoched state time courses (trial time by no. of trials by states)
% T: length of trials (no. trials by 1) in number of time points; all
%    elements of T must equal here
% Y: stimulus or action to test against (no. trials by no. 1; or total time by 1)
% Nperm: number of permutations (default 10000)
% 
% OUTPUTS:
%
% pvalues: a (trial time by 1) vector of pvalues
%
% Diego Vidaurre, University of Oxford (2019)

lambda = 1e-5;  
if nargin < 4, Nperm = 10000; end

if iscell(T)
    if size(T,1) == 1, T = T'; end
    for i = 1:length(T)
        if size(T{i},1)==1, T{i} = T{i}'; end
    end
    T = cell2mat(T);
end

if length(size(Gamma)) == 2
    N = length(T); ttrial = T(1); K = size(Gamma,2);
    if (sum(T) ~= size(Gamma,1))
        error('Gamma has not the right dimensions; use padGamma to adjust?')
    end
    if ~all(T == T(1))
        error('All trials must have the same length; that is, all elements of T must be equal')
    end
    Gamma = reshape(Gamma,[ttrial N K]);
else % length(size(Gamma)) == 3
    N = size(Gamma,2); ttrial = size(Gamma,1); K = size(Gamma,3); 
end

if (size(Y,1) ~= (ttrial*N)) && (length(Y) ~= N)
    error('Dimension of Y not correct');
end
q = size(Y,2);
if q > 1 
    error('Y has to have one column only')
end

if length(Y) ~= N % one value for the entire trial
    for j = 1:N
       if ~all( Y(1,j) == Y(:,j))
           error('Y must have the same value for the entire trial')
       end
    end
    Y = reshape(Y,[ttrial N]);
    Y = Y(1,:);
end
if size(Y,1) < size(Y,2); Y = Y'; end
Y = zscore(Y);
vY = sum(Y.^2);
    
C = zeros(K,N,ttrial);
for t = 1:ttrial 
   g = permute(Gamma(t,:,:),[2 3 1]);
   C(:,:,t) = (g' * g + lambda*eye(K)) \ g' ;
end
    
grotperms = zeros(ttrial,Nperm);
for perm = 1:Nperm
    if perm > 1
        rperm = randperm(N);
        Yp = Y(rperm);
    else
        Yp = Y;
    end
    for t = 1:ttrial
        g = permute(Gamma(t,:,:),[2 3 1]); 
        b = C(:,:,t) * Yp;
        grotperms(t,perm) = 1 - sum((g * b - Yp).^2) / vY; 
    end
end

pvalues = zeros(ttrial,1);
for t = 1:ttrial
    pvalues(t) = sum(grotperms(t,1) <=  grotperms(t,:)) / (Nperm + 1);
end

end

