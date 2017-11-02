function PSD = project_PSD(hmm,Gamma,X,T,A,bands,options)
% Projects a spectral estimation for the hmm from PCA space to original space
%
% Author: Diego Vidaurre, University of Oxford (2016)


ndim = size(A,1);
nbands = size(bands,1);
K = length(hmm.state);

if isfield(hmm.state(1),'f')
    fit = hmm;
else
    fit = hmmspectramar(X,T,hmm,Gamma,options);
end

f = fit.state(1).f; Nf = length(f);
PSD = zeros(nbands,ndim,K);

for k = 1:K
    PSD_kn = zeros(Nf,ndim);
    for fr = 1:Nf
        PSD_fr = diag(diag(permute(fit.state(k).psd(fr,:,:),[2 3 1])));
        for n = 1:ndim
            PSD_kn(fr,n) = A(n,:) * PSD_fr * A(n,:)';
        end
    end
    for j = 1:nbands
        ind = f>=bands(j,1) & f<bands(j,2);
        PSD(j,:,k) = sum(PSD_kn(ind,:));
    end
end



end