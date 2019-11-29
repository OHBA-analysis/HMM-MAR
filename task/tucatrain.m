function [tuda,Gamma,GammaInit,vpath,stats] = tucatrain(X,Y,T,options)
% Wrapper function to perform temporally unconstrained classification
% Analysis (TUCA). This function is now merely a wrapper function with core
% functionality maintained in tudatrain.

[tuda,Gamma,GammaInit,vpath,stats] = tudatrain(X,Y,T,options);
end
