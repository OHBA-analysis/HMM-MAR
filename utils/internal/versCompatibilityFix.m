function hmm = versCompatibilityFix(hmm)

if ~isfield(hmm.train,'orders')
    hmm.train.orders = formorders(hmm.train.order,hmm.train.orderoffset,...
        hmm.train.timelag,hmm.train.exptimelag);
end

if isfield(hmm,'cache')
    hmm = rmfield(hmm,'cache');
end

if isfield(hmm.train,'state')
    hmm.train = rmfield(hmm.train,'state');
end
if ~isfield(hmm.train,'onpower')
    hmm.train.onpower = 0;
end
if ~isfield(hmm.train,'leida')
    hmm.train.leida = 0;
end
if ~isfield(hmm.train,'embeddedlags')
    hmm.train.embeddedlags = 0;
end
if ~isfield(hmm.train,'pca')
    hmm.train.pca = 0;
end
if ~isfield(hmm.train,'filter')
    hmm.train.filter = [];
end
if ~isfield(hmm.train,'detrend')
    hmm.train.detrend = 0;
end
if ~isfield(hmm.train,'downsample')
    hmm.train.downsample = 0;
end
if ~isfield(hmm.train,'leakagecorr')
    hmm.train.leakagecorr = 0;
end
if ~isfield(hmm.train,'Pstructure')
    hmm.train.Pstructure = true(hmm.K);
end
if ~isfield(hmm.train,'Pistructure')
    hmm.train.Pistructure = true(1,hmm.K);
end
if ~isfield(hmm.train,'PriorWeightingP')
    hmm.train.PriorWeightingP = 1;
end
if ~isfield(hmm.train,'PriorWeightingPi')
    hmm.train.PriorWeightingPi = 1;
end
if ~isfield(hmm.train,'lowrank')
    hmm.train.lowrank = 0;
end
if ~isfield(hmm.train,'updateObs')
    hmm.train.updateObs = 1;
end
if ~isfield(hmm.train,'updateGamma')
    hmm.train.updateGamma = 1;
end
if ~isfield(hmm.train,'updateP')
    hmm.train.updateP = 1;
end
if ~isfield(hmm.train,'id_mixture')
    hmm.train.id_mixture = 0;
end
if ~isfield(hmm.train,'id_mixture')
    hmm.train.id_mixture = 0;
end
if ~isfield(hmm.train,'distribution')
    hmm.train.distribution = 'Gaussian';
end
if ~isfield(hmm.train,'pca_spatial')
    hmm.train.pca_spatial = 0;
end
if ~isfield(hmm.train,'Gamma_constraint')
    hmm.train.Gamma_constraint = [];
end

end
