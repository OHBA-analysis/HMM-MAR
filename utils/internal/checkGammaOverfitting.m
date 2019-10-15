function [Gamma,D] = checkGammaOverfitting(Gamma,residuals,hmm,T,orders)
% Check whether a given timecourse is overfitting to the data by assigning
% anticorrelated state timecourses to different conditions. If overfitting
% is occuring for a particular label, that label's timecourse is reset to
% the global mean value of Gamma.
if nargin<5
    orders=0;
end
P=T(1)-orders; 
Ydim = size(residuals,2);
GammaMean(1,:,:)=squeeze(mean(reshape(Gamma(residuals(:,1)~=1,:),P,sum(residuals(:,1)~=1)/P,hmm.train.K),2));
for k=1:Ydim%hmm.train.logisticYdim
    GammaMean(k+1,:,:)=squeeze(mean(reshape(Gamma(residuals(:,k)==1,:),P,sum(residuals(:,k)==1)/P,hmm.train.K),2));
end

%D=dist(GammaMean(:,:)')./size(GammaMean,2);
for k=1:Ydim+1
    for l=k:Ydim+1
        for st = 1:hmm.train.K
            rho = corr([squeeze(GammaMean(k,:,st));squeeze(GammaMean(l,:,st))]');
            Rmat(k,l,st)=rho(2,1);
        end
        D(k,l) = min(Rmat(k,l,:));
    end
end
threshold = 0;
if any(D(1,:)<threshold)
    % Overfitting problem to be recalibrated:
    fprintf('\n WARNING: timecourse is overfitting to labels - resetting now \n');
    fileID = fopen('log.txt','a');
    fprintf(fileID,['\n    Timecourse overfitting to labels; had to recalibrate \n']);
    fclose(fileID);
    if hmm.train.logisticYdim>1
        Gamma0 = squeeze(mean(reshape(Gamma,P,size(Gamma,1)/P,hmm.train.K),2));
        for k=find(D(1,:)<threshold)
            Gamma(residuals(:,k-1)==1,:)=repmat(Gamma0,sum(residuals(:,k-1)==1)/P,1);
        end
    else
        StateLifetimes=squeeze(sum(GammaMean,2));
        [~,toreplace] = min(min(StateLifetimes'));
        tokeep = setdiff([1 2],toreplace);
        Gamma0 = squeeze(GammaMean(toreplace,:,:)); %note order is switched here!
        if toreplace==1
            Gamma0 = squeeze(GammaMean(tokeep,:,:));
            Gamma(residuals(:,1)~=1,:) = repmat(Gamma0,sum(residuals(:,1)~=1)/P,1);
        else
            Gamma0 = squeeze(GammaMean(toreplace,:,:));
            Gamma(residuals(:,1)==1,:) = repmat(Gamma0,sum(residuals(:,1)==1)/P,1);
        end
    end
    
end

end