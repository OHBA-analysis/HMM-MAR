# makefile for hidden_state_inference
# designed for linux distributions
#
# Giles Colclough
#
# Principal Usage:
#
#  make mex
#  make all
#  make clean
#  make docs
#  make tar
#  make tidy - remove object files

# Be aware: Armadillo suggests that linking to blas and lapack should be done STATICALLY
ARMADILLO_LIB = libarmadillo.so

MATLAB_DIR = /opt/fmrib/MATLAB/R2015b/
ARCH       = glnxa64#

# Variables for mex file
TARGET   = hidden_state_inference
SRCS     = ../test/$(TARGET)_test.cpp  ../$(TARGET).cpp
OBJS     = $(TARGET)_test.o $(TARGET).o # $(SRCS:.cpp=.o)
OPT      = -O3 #-O0
CC       = g++ $(OPT) -march=native 
DEBUG    = -DARMA_NO_DEBUG -DNDEBUG # this speeds things up but makes debugging hard. Use -g to help debugging
CFLAGS   = -Wall -c $(DEBUG) 
LFLAGS   = -Wall -larmadillo $(DEBUG) 
INCLUDE  = -I/home/fs0/samh/include -I$(PWD) -I$(PWD)/test -I/usr/include/openblas # armadillo header files are in /usr/include, and armadillo libraries in /usr/lib64
LFLAGS_STATIC = -Wall -framework Accelerate $(DEBUG)

BLAS_LIB     = /home/fs0/samh/lib/libopenblas.so
LAPACK_LIB   = /usr/lib64/liblapack.so
ARPACK_LIB   = /usr/lib64/libarpack.so.2
MWBLAS_LIB   = $(MATLAB_DIR)/bin/$(ARCH)/libmwblas.so
MWLAPACK_LIB = $(MATLAB_DIR)/bin/$(ARCH)/libmwlapack.so

MEX         = $(MATLAB_DIR)/bin/mex
MEX_FLAGS   = -v -largeArrayDims
MEX_INCLUDE = -I/home/fs0/samh/include -I./ -I/usr/include -I/usr/include/openblas
MEX_LFLAGS  = -Wall $(DEBUG) $(ARMADILLO_LIB)
MEX_CXXOPT  = CXXOPTIMFLAGS="$(OPT) $(DEBUG)"

ARMA_STATIC = -DARMA_DONT_USE_WRAPPER # to force linking to lapack and blas, with simple inclusion of header files
ARMA_FLAGS  = -DARMA_BLAS_LONG_LONG -DARMA_64BIT_WORD # These flags are important to allow armadillo to sync with Matlab #-DARMA_USE_CXX11 

MEX_LFLAGS_STATIC = $(DEBUG) CXXLIBS='-Wl,-rpath-link,$(MATLAB_DIR)/bin/$(ARCH) -L$(MATLAB_DIR)/bin/$(ARCH)/ -lmx -lmex -lmat -lm -L/usr/lib66 -L/home/fs0/samh/lib -lopenblas -llapack $(BLAS_LIB) $(LAPACK_LIB) $(ARPACK_LIB)' 
MEX_CXXFLAG       = 




# What to do on calling make with no arguments
default: all


# Multiple target calls
all: mex docs tidy


# Executable targets
test: $(OBJS)
	$(CC) $(OBJS) -o ../test/test $(INCLUDE) $(LFLAGS_STATIC) $(ARMA_FLAGS)

mex: 
	$(MEX) ../$(TARGET)_mx.cpp ../$(TARGET).cpp $(MEX_FLAGS) $(MEX_CXXOPT) $(MEX_INCLUDE) $(MEX_CXXFLAG) $(ARMA_STATIC) $(MEX_LFLAGS_STATIC) $(ARMA_FLAGS)




# Object files
$(TARGET)_test.o: ../$(TARGET).hpp
	$(CC) $(CFLAGS) $(ARMA_FLAGS) $(ARMA_STATIC) ../test/$(TARGET)_test.cpp $(INCLUDE)

$(TARGET).o: 
	$(CC) $(CFLAGS) $(ARMA_FLAGS) $(ARMA_STATIC) ../$(TARGET).cpp $(INCLUDE)


# Utility functions
docs:
	doxygen ../$(TARGET).doxy
	rm -rf ../latex
	rm -rf ../html
	mv -f latex ../
	mv -f html ../
	$(MAKE) -C ../latex

tidy:
	\rm *.o *~ ../*.tar.gz ../*.o ../*~ ../test/*.o ../test/*~

clean:
	\rm -r ../*.o ../linux/*.o ../mac/*.o ../$(TARGET).tar.gz *.mex* ../linux/*.mex* ../mac/*.mex* ../*~ ../linux/*~ ../mac/*~ ../test/*.o ../test/*~ ../test/test ../latex ../html

tar:
	tar cfv ../html_doc.tar.gz ../html/*
	tar cfvz ../$(TARGET).tar.gz makefile ../*.hpp ../*.cpp ../*.mex* ../*.m ../latex/*.pdf ../html_doc.tar.gz ../*.doxy ../linux/*.mex* ../mac/*.mex* ../test/*.cpp ../test/*.hpp ../test/*.m
	rm html_doc.tar.gz
    
#[EOF]
