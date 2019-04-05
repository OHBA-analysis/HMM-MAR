function [psderr,coherr,pcoherr,pdcerr,sdphase] = spectrerr(psdc,pdcc,coh,pcoh,pdc,options,mar)

if nargin<7, mar = 0; end

coherr = []; pcoherr = []; pdcerr = []; sdphase = [];
Nf = options.Nf;
ndim = size(psdc,2);
jksamples = size(psdc,4);
psd = mean(psdc,4);
SA = zeros(Nf,ndim,ndim,jksamples);
if options.to_do(1)==1
    cohA = zeros(Nf,ndim,ndim,jksamples);
    atanhcohA = zeros(Nf,ndim,ndim,jksamples);
    phasefactorA = zeros(Nf,ndim,ndim,jksamples);
end
if options.to_do(2)==1
    atanhpdcA = zeros(Nf,ndim,ndim,jksamples);
    pdcA = zeros(Nf,ndim,ndim,jksamples);
end

for in=1:jksamples
    %fprintf('%d of %d \n',in,size(psdc,4))
    if mar==1
        Si = psdc(:,:,:,in);
    else
        indxk = setdiff(1:jksamples,in);
        Si = mean(psdc(:,:,:,indxk),4);
    end
    SA(:,:,:,in) = Si;
    if options.to_do(1)==1
        cohi = zeros(Nf,ndim,ndim);
        atanhcohi = zeros(Nf,ndim,ndim);
        phasefactori = zeros(Nf,ndim,ndim);
        for j = 1:ndim
            for l = 1:ndim
                cjl = Si(:,j,l)./sqrt(Si(:,j,j) .* Si(:,l,l));
                cohi(:,j,l) = abs(cjl);
                atanhcohi(:,j,l) = sqrt(2*jksamples-2)*atanh(cohi(:,j,l));
                phasefactori(:,j,l)=cjl./cohi(:,j,l);
            end
        end
        cohA(:,:,:,in) = cohi;
        atanhcohA(:,:,:,in) = atanhcohi;
        phasefactorA(:,:,:,in) = phasefactori;
    end
    if options.to_do(2)==1
        atanhpdci = zeros(Nf,ndim,ndim);
        if isempty(pdcc)
            pdci = subrutpdc(Si,options.numIterations,options.tol);
        else
            pdci = pdcc(:,:,:,in);
        end
        for j = 1:ndim
            for l = 1:ndim
                atanhpdci(:,j,l) = sqrt(2*jksamples-2)*atanh(pdci(:,j,l));
            end
        end
        atanhpdcA(:,:,:,in) = atanhpdci;
        pdcA(:,:,:,in) = pdci;
    end
end
dof = jksamples;
tcrit = tinv(1-options.p/2,dof-1); % Inverse of Student's T cumulative distribution function
% psd errors
sigmaS = tcrit * sqrt(dof-1) * std(log(SA),1,4);
psderr = zeros([2 size(psd)]);
psderr(1,:,:,:) = psd .* exp(-sigmaS);
psderr(2,:,:,:) = psd .* exp(sigmaS);
% coh errors
dof = 2*jksamples;
tcrit = tinv(1-options.p/2,dof-1);
if (options.to_do(1)==1)
    atanhcoh = sqrt(2*jksamples-2)*atanh(coh); % z
    sigma12 = sqrt(jksamples-1)*std(atanhcohA,1,4); % Jackknife estimate std(z)=sqrt(dim-1)*std of 1-drop estimates
    Cu = atanhcoh+tcrit*sigma12;
    Cl = atanhcoh-tcrit*sigma12;
    coherr = zeros([2 size(coh)]);
    coherr(1,:,:,:) = max(tanh(Cl/sqrt(2*jksamples-2)),0); % This ensures that the lower confidence band remains positive
    coherr(2,:,:,:) = tanh(Cu/sqrt(2*jksamples-2));
    % pcoh errors
    atanhcoh = sqrt(2*jksamples-2)*atanh(pcoh); % z
    sigma12 = sqrt(jksamples-1)*std(atanhcohA,1,4); % Jackknife estimate std(z)=sqrt(dim-1)*std of 1-drop estimates
    Cu = atanhcoh+tcrit*sigma12;
    Cl = atanhcoh-tcrit*sigma12;
    pcoherr = zeros([2 size(pcoh)]);
    pcoherr(1,:,:,:) = max(tanh(Cl/sqrt(2*jksamples-2)),0); % This ensures that the lower confidence band remains positive
    pcoherr(2,:,:,:) = tanh(Cu/sqrt(2*jksamples-2));
    % std of the phase
    sdphase = sqrt( (2*jksamples-2)*(1-abs(mean(phasefactorA,4))) );
end
% pdc errors
if (options.to_do(2)==1)
    atanhpdc = sqrt(2*jksamples-2)*atanh(pdc); % z
    sigma12 = sqrt(jksamples-1)*std(atanhpdcA,1,4); % Jackknife estimate std(z)=sqrt(dim-1)*std of 1-drop estimates
    Cu = atanhpdc+tcrit*sigma12;
    Cl = atanhpdc-tcrit*sigma12;
    pdcerr = zeros([2 size(pdc)]);
    pdcerr(1,:,:,:) = tanh(Cl/sqrt(2*jksamples-2)); % This ensures that the lower confidence band remains positive
    pdcerr(2,:,:,:) = tanh(Cu/sqrt(2*jksamples-2));
end

end
