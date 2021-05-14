UPDATING TO THE LATEST VERSION IS STRONGLY RECOMMENDED, DUE TO AN ERROR THAT WAS INTRODUCED AT THE END OF JANUARY 2021, AND CLANED AT THE END OF MARCH 2021.

**HMM-MAR**

Please check the Wiki for the latest documentation, including some basic introduction to the method. If you have issues of questions, it is possible to email us, but please better use the Issues tab on github, so that others can see the response as well.

Note: the name of the toolbox is mantained only for historical reasons, and currently contains other observation models other than the multivariate autoregressive model (MAR).

The example scripts provide some basic demonstration of the toolbox functionality. The script examples/run_HMMMAR.m is a template script that specifies some basic options depending on the specified data modality and, provided that the user has already loaded the data in the right format (see the script for details), runs the HMM-MAR and gets some basic information out of the estimation. The script examples/run_HMMMAR_2.m is a template script that generates some random (putatively fMRI) data under two different conditions, estimates the HMM for a number of different configurations, does permutation testing to test the results against the conditions, and plots the results.

Under _examples/_, there are scripts demonstrating the analysis conducted for some of the papers

For more detailed description and applications, please refer to

If this toolbox turns out to be useful, we'd grateful if you cite the main references for the HMM-MAR: 

> _Diego Vidaurre, Andrew J. Quinn, Adam P. Baker, David Dupret, Alvaro Tejero-Cantero and Mark W. Woolrich (2016) [Spectrally resolved fast transient brain states in electrophysiological data](http://www.sciencedirect.com/science/article/pii/S1053811915010691). **NeuroImage**. Volume 126, Pages 81–95._

and, describing an efficient inference (stochastic) method for big amounts of data, 

> _Diego Vidaurre, R. Abeysuriya, R. Becker, Andrew J. Quinn, F. Alfaro-Almagro, S.M. Smith and Mark W. Woolrich (2017) [Discovering dynamic brain networks from Big Data in rest and task](http://www.sciencedirect.com/science/article/pii/S1053811917305487). **NeuroImage**._

An example of application on fMRI is shown in 

> _Diego Vidaurre, S.M. Smith and Mark W. Woolrich (2017). [Brain network dynamics are hierarchically organized in time](http://www.pnas.org/content/early/2017/10/26/1705120114). **Proceedings of the National Academy of Sciences of the USA**_

A version adequate for modelling whole-brain M/EEG data (not MAR-based, but using lagged cross-correlations) is proposed in 

> _Diego Vidaurre, Lawrence T. Hunt, Andrew J. Quinn, Benjamin A.E. Hunt, Matthew J. Brookes, Anna C. Nobre and Mark W. Woolrich (2017). [Spontaneous cortical activity transiently organises into frequency specific phase-coupling networks](https://www.nature.com/articles/s41467-018-05316-z). **Nature Communications**._

A step-by-step paper detailing the use of the HMM for MEG alongside comprehensive details of MEG preprocessing in 

> _Andrew J. Quinn, Diego Vidaurre, Romesh Abeysuriya, Robert Becker, Anna C Nobre, Mark W Woolrich (2018). [Task-Evoked Dynamic Network Analysis Through Hidden Markov Modeling](https://www.frontiersin.org/articles/10.3389/fnins.2018.00603/full). **Frontiers in Neuroscience**._

An HMM-based model to find dynamic decoding models, where the states define how, when and where the stimulus is encoded in the brain

> Diego Vidaurre, Nicholas Myers, Mark Stokes, Anna C Nobre and Mark W. Woolrich (2018). [Temporally unconstrained decoding reveals consistent but time-varying stages of stimulus processing](https://academic.oup.com/cercor/article/29/2/863/5232535). **Cerebral Cortex**.


