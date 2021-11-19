function metricsout = assess_sign_flips(netmats,flips)
% function for comparing different sign flip solutions
if min(unique(flips))==0
    flips = 1-2*flips;
end
[N,nch] = size(flips);
nlags = size(netmats,1)/nch;
for iSj=1:N
    flips_to_do = repelem(flips(iSj,:),1,nlags);
    flipped_netmats(iSj,:,:) = netmats(:,:,iSj).*(flips_to_do'*flips_to_do);
end
offdiagblocks = triu(ones(nlags*nch)-repelem(eye(nch),nlags,nlags));
values = flipped_netmats(:,logical(offdiagblocks));

metricsout = [];
metricsout.abscorr = mean(abs(sum(values)));
metricsout.corr = corr(values');
metricsout.corrmean = sum(sum(triu(metricsout.corr,1)))./((N.^2-N)/2);
end