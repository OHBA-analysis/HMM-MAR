function [pvals,pvalsFWE,pvalmat] = permtestmass_NPC(Yin,Xin,Nperm,Perms,conf,...
    groupingYin,groupingXin,verbose)
% 
% It tests each replication (column) of Yin vs each variable (column) of Xin, 
% combining the tests across replications such that we have an aggregated  
% final p-value as well as specific p-value for each column of Yin.
%
% ARGUMENTS
% - Yin: Noisy replications
% - Xin: Observed behavioural variables.
% - Nperm: is the number of permutations
% - Perms: Precomputed permutations (e.g. using Palm) 
% - conf: the confounds we want to regress out
% - Perms: precomputed permutations, (Nsubjects by Nperm)
% - groupingY: grouping of variables in Yin, by default no grouping
% - groupingX: grouping of variables in Xin, by default no grouping;
%           example: [1 1 1 2 3 3 3 3 4 4] 
%
% - pvals contains 1 p-value per column in Xin;
% - pvalsFWE contains 1 p-value per column in Xin, familywise error rate corrected
% - pvalmat contains the p x q matrix of pvalues for the first outer loop
%       interation (unpermuted)
%
% Diego Vidaurre, University of Oxford (2017)

if nargin<4 
    Perms = [];
end
if nargin<5 
    conf = [];
end
if nargin<6 || isempty(groupingYin)
    groupingYin = 1:size(Yin,2);
end
if nargin<7 || isempty(groupingXin)
    groupingXin = 1:size(Xin,2);
end
if nargin<8
    verbose = 0;
end

N = size(Xin,1);
R = length(unique(groupingYin));
P = length(unique(groupingXin));

indexes_Y = cell(R,1); L_Y = zeros(R,1); 
for j = 1:R
   indexes_Y{j} = find(groupingYin==j);
   L_Y(j) = length(indexes_Y{j});
end
indexes_X = cell(P,1); L_X = zeros(P,1);
for j = 1:P
   indexes_X{j} = find(groupingXin==j);
   L_X(j) = length(indexes_X{j});
end

is_binary = false(P,1);
for j = 1:P
    if L_X(j)>1, continue; end
    vals = unique(Xin(:,indexes_X{j}));
    is_binary(j) = length(vals) == 2;
    if is_binary(j)
       y = Xin(:,indexes_X{j}); 
       Xin(y==vals(1),indexes_X{j}) = 1; Xin(y==vals(2),indexes_X{j}) = 2;  
    end
end

if ~isempty(conf)
    conf = conf - repmat(mean(conf),N,1);
    Yin = Yin - repmat(mean(Yin),N,1);
    Yin = Yin - conf * pinv(conf) * Yin;
    %if any(~is_binary)
    %    Yin(:,~is_binary) = Yin(:,~is_binary) - repmat(mean(Yin(:,~is_binary)),N,1);
    %    Yin(:,~is_binary) = Yin(:,~is_binary) - conf * pinv(conf) * Yin(:,~is_binary);
    %end
end

Yin = zscore(Yin);
if any(~is_binary)
    Xin(:,~is_binary) = zscore(Xin(:,~is_binary));
end

T = zeros(Nperm,P);

% Unpermuted
pv = zeros(R,P);
for j = 1:P
    for i = 1:R
        if is_binary(j) && L_Y(i)==1
            [~,pv(i,j)] = ttest2(Yin(Xin(:,indexes_X{j})==1,indexes_Y{i}), ...
                Yin(Xin(:,indexes_X{j})==2,indexes_Y{i}));
        elseif L_Y(i)==1 && L_X(j)==1
            [~,pv0] = corrcoef(Yin(:,i),Xin(:,j));
            pv(i,j) = pv0(1,2);
        elseif L_Y(i)>1 && L_X(j)==1
            [~,~,~,~,stats] = regress(Xin(:,indexes_X{j}),[ones(N,1) Yin(:,indexes_Y{i})]); 
            pv(i,j) = stats(3);
        elseif L_Y(i)==1 && L_X(j)>1
            [~,~,~,~,stats] = regress(Yin(:,indexes_Y{i}),[ones(N,1) Xin(:,indexes_X{j})]); 
            pv(i,j) = stats(3);            
        else
            [~,~,~,~,~,stats] = canoncorr(Yin(:,indexes_Y{i}),Xin(:,indexes_X{j}));
            pv(i,j) = stats.p(1); 
        end
    end
end
T(1,:) = -2 * sum(log(pv));
pvalmat = pv;
if verbose, disp('1'); end

% Permutation loop
X = Xin; 
parfor i1 = 2:Nperm
    pv = zeros(R,P);
    if isempty(Perms)
        perm = randperm(N);
    else
        perm = Perms(:,randperm(size(Perms,1),1));
    end
    Xin = X(perm,:);
    for j = 1:P
        for i = 1:R
            if is_binary(j) && L_Y(i)==1
                [~,pv(i,j)] = ttest2(Yin(Xin(:,indexes_X{j})==1,indexes_Y{i}), ...
                    Yin(Xin(:,indexes_X{j})==2,indexes_Y{i}));
            elseif L_Y(i)==1 && L_X(j)==1
                [~,pv0] = corrcoef(Yin(:,i),Xin(:,j));
                pv(i,j) = pv0(1,2);
            elseif L_Y(i)>1 && L_X(j)==1
                [~,~,~,~,stats] = regress(Xin(:,indexes_X{j}),[ones(N,1) Yin(:,indexes_Y{i})]);
                pv(i,j) = stats(3);
            elseif L_Y(i)==1 && L_X(j)>1
                [~,~,~,~,stats] = regress(Yin(:,indexes_Y{i}),[ones(N,1) Xin(:,indexes_X{j})]);
                pv(i,j) = stats(3);
            else
                [~,~,~,~,~,stats] = canoncorr(Yin(:,indexes_Y{i}),Xin(:,indexes_X{j}));
                pv(i,j) = stats.p(1);
            end
        end
    end
    T(i1,:) = -2 * sum(log(pv(:,:,1)));
    if verbose, disp(num2str(i1)); end
end

Tmax = repmat(max(T,[],2),1,P);
pvals = sum( repmat(T(1,:),Nperm,1) <= T ) ./ (1 + Nperm) ;
pvalsFWE = sum( repmat(T(1,:),Nperm,1) <= Tmax ) ./ (1 + Nperm) ;

end


