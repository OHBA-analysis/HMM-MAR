function encmodel = tudaencoding(X,Y,T,options,Gamma)
% Compute maps representing the "encoding" model for each
% state from the TUDA model. There are two approaches for achieving this:
% For each state, 
% if options.CCA == 1,
%   for each sensor/voxel, the map will have the canonical correlation
%   between the (time by stimulus features) stimulus matrix Y
%   and the sensor/voxel time series. 
% if options.CCA == 0, (default)
%   for each sensor/voxel, the map will have the explained variance from
%   regressing Y on the data at this sensor/voxel. 
% 
% Author: Diego Vidaurre, OHBA, University of Oxford (2017)


if ~isfield(options,'embeddedlags'), embeddedlags = 0;
else, embeddedlags = options.embeddedlags; end
do_embedding = length(embeddedlags)>1;
if ~isfield(options,'CCA'), CCA = 0;
else, CCA = options.CCA; end
nlags = length(embeddedlags);

options.Nfeatures = 0; 
options.pca = 0;
options.embeddedlags = 0;

[X,Y,T] = preproc4hmm(X,Y,T,options); 

K = size(Gamma,2);
p = size(X,2);
q = size(Y,2);
encmodel = zeros(p,K);

for j = 1:p
    if do_embedding
        Xj = embeddata(X(:,j),T,embeddedlags);
    else
        Xj = X(:,j);
    end
    for k = 1:K
        if CCA
            G = repmat(Gamma(:,k),1,nlags);
            m = sum(G .* Xj) / sum(Gamma(:,k));
            Xw = sqrt(G) .* (Xj - repmat(m,size(G,1),1));
            G = repmat(Gamma(:,k),1,q);
            m = sum(G .* Y) / sum(Gamma(:,k));
            Yw = sqrt(G) .* (Y - repmat(m,size(G,1),1));
            [~,~,r] = canoncorr(Xw,Yw); % weighted CCA
            encmodel(j,k) = sum(r);
        else       
            G1 = repmat(Gamma(:,k),1,nlags);
            Xw = sqrt(G1) .* Xj;
            G2 = repmat(Gamma(:,k),1,q);
            Yw = sqrt(G2) .* Y;
            beta = (Yw' * Yw) \ (Yw' * Xw);
            res = ((Xj - Y * beta).^2) .* G1; 
            res0 = (Xj.^2) .* G1;
            encmodel(j,k) = 1 - sum(res(:))/sum(res0(:)); 
        end
    end
end

end