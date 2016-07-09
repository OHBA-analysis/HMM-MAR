function [A,B] = highdim_pca(X,T,d,embeddedlags,standardise)
% pca for potentially loads of subjects
% if X is a cell of things, uses SVD
% if X is a matrix, uses Matlab's PCA 

if nargin<3, embeddedlags = 0; end
if nargin<4, standardise = 1; end

is_cell_strings = iscell(X) && ischar(X{1});
is_cell_matrices = iscell(X) && ~ischar(X{1});
options = struct();
options.standardise = standardise;
options.embeddedlags = embeddedlags;
options.pca = 0; % PCA is done here! 

if is_cell_strings || is_cell_matrices
    B = [];
    for i=1:length(X)
        X_i = loadfile(X{i},T{i},options); % embedded is done here
        X_i = X_i - repmat(mean(X_i),size(X_i,1),1); % must center
        if i==1, C = zeros(size(X_i,2)); end
        C = C + X_i' * X_i;
    end
    [A,~,~] = svd(C); A = A(:,1:d);
else
    if length(embeddedlags)>1
       X = embeddata(X,T,options.embeddedlags); 
    end
    [A,B] = pca(X,'NumComponents',d,'Centered',true);  
end

end