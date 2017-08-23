function [V,D] = globaleig(X,T,embeddedlags,standardise,onpower,detrend,filter,downsample,Fs)
% Global eigendecomposition
%
% Author: Diego Vidaurre, University of Oxford (2017)

if nargin<3, embeddedlags = 0; end
if nargin<4, standardise = 0; end
if nargin<5, onpower = 0; end
if nargin<6, detrend = 0; end
if nargin<7, filter = []; end
if nargin<8, downsample = 0; end
if nargin<9, Fs = 1; end

options = struct();
options.filter = filter; 
options.Fs = Fs; 
options.standardise = standardise;
options.embeddedlags = embeddedlags;
options.pca = 0;
options.onpower = onpower;
options.detrend = detrend;
options.downsample = downsample;
options.firsteigv = 1; 
options.As = [];  

Tsum = 0; 
for i=1:length(X)
    X_i = loadfile(X{i},T{i},options); % zscoring/embeddeding/centering are done here
    if i==1, C = zeros(size(X_i,2)); end
    C = C + X_i' * X_i;
    Tsum + Tsum + T{i};
end
[V,D] = svd(C);
D = diag(D);

end