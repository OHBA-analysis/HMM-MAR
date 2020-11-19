function MLGamma = gammaRasterPlot(Gamma,T_hmm,t_axispoints,t_axislabels)
% plotting function that takes a state time course Gamma, reshapes it into
% trials given by T (which must be uniform), and plots a raster plot of the
% max likelihood state vs trials.

[NT,K] = size(Gamma);
T = T_hmm(1);
if size(T_hmm,2)>1
    nTr = length(T_hmm);
else
    nTr = NT/T;
end

MLGamma = Gamma==repmat(max(Gamma,[],2),1,K);
MLGamma = mod(find(MLGamma'),K);
MLGamma(MLGamma==0) = K;

rastimage = reshape(MLGamma,T,nTr);
imagesc(rastimage');
ylabel('Trials','fontsize',16);
if nargin>3
    set(gca,'XTick',t_axispoints)
    set(gca,'XTickLabel',t_axislabels,'fontsize',16);
    xlabel('Time')
end
set(gca,'fontsize',16);

end