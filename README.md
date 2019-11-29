### HMM-MAR

Please check the [Wiki](https://github.com/OHBA-analysis/HMM-MAR/wiki) for the latest documentation, including some basic introduction to the method. 

Note: the name of the toolbox is mantained only for historical reasons, and currently contains other observation models other than the multivariate autoregressive model (MAR). 

The example scripts provide some basic demonstration of the toolbox functionality. The script examples/run_HMMMAR.m is a template script that specifies some basic options depending on the specified data modality and, provided that the user has already loaded the data in the right format (see the script for details), runs the HMM-MAR and gets some basic information out of the estimation. The script examples/run_HMMMAR_2.m is a template script that generates some random (putatively fMRI) data under two different conditions, estimates the HMM for a number of different configurations, does permutation testing to test the results against the conditions, and plots the results. 

Under examples/, there are scripts demonstrating the analysis conducted for the papers: Vidaurre et al. (2016) NeuroImage, Vidaurre et al. (2017) PNAS, Vidaurre et al. (2018) Nature Communications

For more detailed description and applications, please refer to 

Diego Vidaurre, Andrew J. Quinn, Adam P. Baker, David Dupret, Alvaro Tejero-Cantero and Mark W. Woolrich (2016) _Spectrally resolved fast transient brain states in electrophysiological data._ NeuroImage. Volume 126, Pages 81–95.

Diego Vidaurre, Romesh Abeysuriya, Robert Becker, Andrew J. Quinn, F. Alfaro-Almagro, S.M Smith and Mark W. Woolrich (2018) _Discovering dynamic brain networks from big data in rest and task._ NeuroImage.    

Diego Vidaurre, S.M. Smith and Mark W. Woolrich (2017). _Brain network dynamics are hierarchically organized in time_. Proceedings of the National Academy of Sciences of the USA.

Diego Vidaurre, Lawrence T. Hunt, Andrew J. Quinn, Benjamin A.E. Hunt, Matthew J. Brookes, Anna C. Nobre and Mark W. Woolrich (2018). _Spontaneous cortical activity transiently organises into frequency specific phase-coupling networks_. Nature Communications.

Andrew J. Quinn, Diego Vidaurre, Romesh Abeysuriya, Robert Becker, Anna C Nobre, Mark W Woolrich (2019). _Task-Evoked Dynamic Network Analysis Through Hidden Markov Modeling_. Frontiers in Neuroscience.
