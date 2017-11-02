function hmmproj = project_hmm(hmm,A,B,V)
% Projects an hmm estimation from PCA space to original space
% A : standard PCA
% B : PCA on MAR (per order)
% V : PCA on regressors (all orders)
% using the matrix A/B/V 
% it projects the HMM-MAR parameters to the space spanned by A/B/V
% it ignores the covariance of W, just operates on the mean (I need to implement this) 
%
% Author: Diego Vidaurre, University of Oxford (2016)

if nargin<4, V = []; end
if nargin<3, B = []; end
if nargin<2, A = []; end

K = length(hmm.state);
zeromean = hmm.train.zeromean; 

if ~isempty(B) && ~isempty(V)
    error('Only one of B/V can be non-empty')
end

% if ndim==0
%     warning('Only the covariance matrix was modelled - nothing to do')
%     return
% end

hmmproj = hmm;

% PCA on the MAR
if ~isempty(B) % assumes centered data
    [ndim,ndimpca] = size(A);
    for k = 1:K
        W = zeros(ndim*p+(~zeromean),ndim);
        if ~zeromean
            W(1,:) = hmm.state(k).W.Mu_W(1,:);
        end
        for n = 1:ndim
            for j=1:p
                ind = (1:ndim) + ndim*(j-1) +(~zeromean);
                indpca = (1:ndimpca) + ndim*(j-1) +(~zeromean);
                W(ind,n) = hmm.state(k).W.Mu_W(indpca,n) * B';  
            end
        end
        hmmproj.state(k).W.Mu_W = W; 
        if isfield(hmmproj.state(k).W,'S_W')
            hmmproj.state(k).W = rmfield(hmmproj.state(k).W,'S_W');
        end
        if isfield(hmmproj.state(k).W,'iS_W')
            hmmproj.state(k).W = rmfield(hmmproj.state(k).W,'iS_W');
        end
    end
end

% PCA on regressors 
if ~isempty(V) % assumes centered data
    ndim = size(hmm.state(1).W.Mu_W,2);
    p = length(formorders(hmm.train.order,hmm.train.orderoffset,...
        hmm.train.timelag,hmm.train.exptimelag));
    for k = 1:K
        W = zeros(ndim*p+(~zeromean),ndim);
        if ~zeromean
            W(1,:) = hmm.state(k).W.Mu_W(1,:);
        end
        for n = 1:ndim
            W(1+(~zeromean):end,n) = V * hmm.state(k).W.Mu_W(1+(~zeromean):end,n);
        end
        hmmproj.state(k).W.Mu_W = W;
        if isfield(hmmproj.state(k).W,'S_W')
            hmmproj.state(k).W = rmfield(hmmproj.state(k).W,'S_W');
        end
        if isfield(hmmproj.state(k).W,'iS_W')
            hmmproj.state(k).W = rmfield(hmmproj.state(k).W,'iS_W');
        end
    end
end

% standard PCA
if ~isempty(A) % assumes centered data
    p = (size(hmm.state(1).W.Mu_W,1)-(~zeromean)) / size(hmm.state(1).W.Mu_W,2);
    [ndim,ndimpca] = size(A);
    for k = 1:K
        W = zeros(ndim*p+(~zeromean),ndim);
        if ~zeromean
            W(1,:) = hmm.state(k).W.Mu_W(1,:) * A'; 
        end
        for j=1:p
            ind = (1:ndim) + ndim*(j-1) +(~zeromean);
            indpca = (1:ndimpca) + ndim*(j-1) +(~zeromean);
            W(ind,:) = A * hmm.state(k).W.Mu_W(indpca,:) * A'; 
        end
        hmmproj.state(k).W.Mu_W = W;
        if isfield(hmmproj.state(k).W,'S_W')
            hmmproj.state(k).W = rmfield(hmmproj.state(k).W,'S_W');
        end
        if isfield(hmmproj.state(k).W,'iS_W')
            hmmproj.state(k).W = rmfield(hmmproj.state(k).W,'iS_W');
        end
    end
end

    

end