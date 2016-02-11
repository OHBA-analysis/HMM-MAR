=Hidden State Inference=
==A mex implementation of the forward-backward algorithm for VB-HMMs==

=== Giles Colclough, OHBA, Feb 2016 ===
----------------------------------------------------------------------

This package provides c++ files for performing the forward-backward algorithm in VB HMM inference. 

The code can be compiled as a c++ function, or a mex file. 
Pre-compiled mex files are provided for osx and glnxa64. If you have trouble, try compiling yourself. 

- If you have openblas, lapack and arpack, this is as simple as typing 'make mex' from the relevant folder. 
- If you need more help, check out installing-on-linux.txt in the linux folder. 

Documentation is compiled by typing 'make docs' from either the linux or mac folders. 


Have a look at the make files, or get in touch for more assistance. 

Giles.Colclough@ohba.ox.ac.uk


NB: You do need openblas, arpack and lapack on Linux. Mac OS should be fine as it is. 



