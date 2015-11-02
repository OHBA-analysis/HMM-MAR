function [psderr,coherr,pcoherr,pdcerr,sdphase] = spectrerr(psdc,pdcc,coh,pcoh,pdc,options)

coherr = []; pcoherr = []; pdcerr = []; sdphase = [];
Nf = options.Nf;
ndim = size(psdc,2);
jksamples = size(psdc,4);
psd = mean(psdc,4);
SA = []; cohA = []; pdcA = [];  
atanhcohA = [];
atanhpdcA = [];
phasefactorA = [];
for in=1:jksamples;
    %fprintf('%d of %d \n',in,size(psdc,4))
    indxk=setdiff(1:jksamples,in);
    Si = mean(psdc(:,:,:,indxk),4);
    SA = cat(4,SA,Si);
    if (options.to_do(1)==1),
        cohi = zeros(Nf,ndim,ndim);
        atanhcohi = zeros(Nf,ndim,ndim);
        atanhpdci = zeros(Nf,ndim,ndim);
        phasefactori = zeros(Nf,ndim,ndim);
        for j=1:ndim,
            for l=1:ndim,
                cjl = Si(:,j,l)./sqrt(Si(:,j,j) .* Si(:,l,l));
                cohi(:,j,l) = abs(cjl);
                atanhcohi(:,j,l) = sqrt(2*jksamples-2)*atanh(cohi(:,j,l));
                phasefactori(:,j,l)=cjl./cohi(:,j,l);
            end
        end
        cohA = cat(4,cohA,cohi);
        atanhcohA = cat(4,atanhcohA,atanhcohi);
        phasefactorA = cat(4,phasefactorA,phasefactori);
    end
    if (options.to_do(2)==1),
        if isempty(pdcc)
            pdci = subrutpdc(Si,options.numIterations,options.tol);
        else
            pdci = pdcc(:,:,:,in);
        end
        for j=1:ndim,
            for l=1:ndim,
                atanhpdci(:,j,l) = sqrt(2*jksamples-2)*atanh(pdci(:,j,l));
            end
        end
        atanhpdcA = cat(4,atanhpdcA,atanhpdci);
        pdcA = cat(4,pdcA,pdci);
    end
end
dof = jksamples-1;
tcrit=tinv(1-options.p/2,dof); % Inverse of Student's T cumulative distribution function
% psd errors
sigmaS = tcrit * sqrt(dof)*std(log(SA),1,4);
psderr = zeros([2 size(psd)]);
psderr(1,:,:,:) = psd .* exp(-sigmaS);
psderr(2,:,:,:) = psd .* exp(sigmaS);
% coh errors
if (options.to_do(1)==1),
    atanhcoh=sqrt(2*jksamples-2)*atanh(coh); % z
    sigma12=sqrt(jksamples-1)*std(atanhcohA,1,4); % Jackknife estimate std(z)=sqrt(dim-1)*std of 1-drop estimates
    Cu=atanhcoh+tcrit*sigma12;
    Cl=atanhcoh-tcrit*sigma12;
    coherr = zeros([2 size(coh)]);
    coherr(1,:,:,:) = max(tanh(Cl/sqrt(2*jksamples-2)),0); % This ensures that the lower confidence band remains positive
    coherr(2,:,:,:) = tanh(Cu/sqrt(2*jksamples-2));
    % pcoh errors
    atanhcoh=sqrt(2*jksamples-2)*atanh(pcoh); % z
    sigma12=sqrt(jksamples-1)*std(atanhcohA,1,4); % Jackknife estimate std(z)=sqrt(dim-1)*std of 1-drop estimates
    Cu=atanhcoh+tcrit*sigma12;
    Cl=atanhcoh-tcrit*sigma12;
    pcoherr = zeros([2 size(pcoh)]);
    pcoherr(1,:,:,:) = max(tanh(Cl/sqrt(2*jksamples-2)),0); % This ensures that the lower confidence band remains positive
    pcoherr(2,:,:,:) = tanh(Cu/sqrt(2*jksamples-2));
    % std of the phase
    sdphase = sqrt( (2*jksamples-2)*(1-abs(mean(phasefactorA,4))) );
end
% pdc errors
if (options.to_do(2)==1),
    atanhpdc=sqrt(2*jksamples-2)*atanh(pdc); % z
    sigma12=sqrt(jksamples-1)*std(atanhpdcA,1,4); % Jackknife estimate std(z)=sqrt(dim-1)*std of 1-drop estimates
    Cu=atanhpdc+tcrit*sigma12;
    Cl=atanhpdc-tcrit*sigma12;
    pdcerr = zeros([2 size(pdc)]);
    pdcerr(1,:,:,:) = max(tanh(Cl/sqrt(2*jksamples-2)),0); % This ensures that the lower confidence band remains positive
    pdcerr(2,:,:,:) = tanh(Cu/sqrt(2*jksamples-2));
end
