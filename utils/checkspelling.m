function options_checked = checkspelling (options)

potential_options = ... 
    {'K','order','covtype','zeromean','embeddedlags',... % stndard options
    'timelag','exptimelag','orderoffset','symmetricprior','uniqueAR','S','prior',...
    'state','Fs','cyc','tol','meancycstop','cycstogoafterevent','DirStats',...
    'initcyc','initrep','inittype','Gamma','hmm','fehist','DirichletDiag',...
    'dropstates','repetitions','updateGamma','decodeGamma','keepS_W',...
    'useParallel','useMEX','verbose','cvfolds','cvrep','cvmode','cvverbose',...
    'BIGNbatch','BIGuniqueTrans','BIGprior','BIGcyc','BIGmincyc',... % stochastic
    'BIGundertol_tostop','BIGcycnobetter_tostop','BIGtol','BIGinitrep','BIGdecodeGamma',...
    'BIGforgetrate','BIGdelay','BIGbase_weights','BIGcomputeGamma','BIGverbose',...
    'p','removezeros','completelags','rlowess','numIterations','tol',... % spectra
    'pad','Fs','fpass','tapers','win','to_do','loadings','Nf','MLestimation','level'};

current_options = fieldnames(options);
options_checked = options;

for i=1:length(current_options)
   opt = current_options{i}; 
   found = 0;
   for j=1:length(potential_options)
      if strcmp(opt,potential_options{j}), found = 1; break; end
   end
   if ~found
       warning(sprintf('%s is not a valid option (maybe misspelled?)',opt))
       options_checked = rmfield(options_checked,opt);
   end
end

end