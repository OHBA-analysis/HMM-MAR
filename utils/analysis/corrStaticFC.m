function C = corrStaticFC (data,T,toPlot)
% When running the HMM on fMRI data (with order=0, i.e. a Gaussian
% distribution with mean and covariance), 
% it can happen that states are assigned to entire subjects
% with not much switching between states. This happens typically when the
% static functional connectivity (FC) is so different that states specialise 
% into specific subjects, explaining these grand patterns with
% no room to capture any dynamic FC. 
% This function computes the (subjects by subjects) matrix of static FC 
% similarities (measured in terms of correlation) between each pair of
% subjects. If the obtained values are too low, then covtype='uniquefull'
% has a higher chance to do a good job. 
% 
% Author: Diego Vidaurre (2017)

if nargin<3, toPlot = 1; end
if iscell(data), N = length(data);  
else, N = length(T);  
end

for j = 1:N
    if iscell(data)
        if ischar(data{j})
            dat = load(data{j});
            if isfield(dat,'X')
                X = dat.X;
            else
                X = getfield(dat,char(fieldnames(dat)));
            end
        else
            X = data{j};
        end
    else
        ind = (1:T(j)) + sum(T(1:j-1));
        X = data(ind,:);
    end
    if j==1
        ndim = size(X,2);
        FC = zeros(ndim*(ndim-1)/2,N);
    end
    X = zscore(X); 
    c = corr(X);
    FC(:,j) =  c(triu(true(ndim),1))';
end

C = corr(FC);
C(eye(N)==1) = Inf;
if toPlot
    imagesc(C,[-1 1]); colorbar
    set(gca,'FontSize',18)
    colormap('jet');
    grotc=colormap;  grotc(end,:)=[.8 .8 .8];  colormap(grotc);
end

end
