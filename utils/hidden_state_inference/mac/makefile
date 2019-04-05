# makefile for hidden_state_inference
#
# Giles Colclough
#
# Principal Usage:
#
#  make test
#  make mex
#  make all
#  make clean
#  make docs
#  make tar
#  make tidy - remove object files

# USER NOTES: CHANGE THE LINES BELOW TO REFLECT YOUR MATLAB INSTALLATION
MATLAB_DIR=/Applications/MATLAB_R2016b.app/#
MATLAB_ARCH=maci64# e.g. $(MATLAB_DIR)/bin/$(MATLAB_ARCH)/
MACOS_VERSION=10.11

# If you want to link dynamically, change the lines below to reflect your armadillo installation
# Install armadillo on mac with 'brew install armadillo' (assuming you have homebrew)
#
# If you don't have homebrew, get it by running this at the terminal prompt.
# /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"
#
# Matlab mex files need these direct paths to the dynamic libraries. If you know how to link statically, please tell me. 
ARMADILLO_DIR=/usr/local/Cellar/armadillo/6.500.5
ARMADILLO_LIB=$(ARMADILLO_DIR)/libarmadillo.dylib #file found in ARMADILLO_DIR





# Variables for mex file
TARGET   = hidden_state_inference
SRCS     = ../test/$(TARGET)_test.cpp  ../$(TARGET).cpp
OBJS     = $(TARGET)_test.o $(TARGET).o # $(SRCS:.cpp=.o)
OPT      = -Ofast # -O0 #
CC       = clang++ $(OPT) -march=native -std=c++14 
DEBUG    = -DARMA_NO_DEBUG -DNDEBUG # this speeds things up but makes debugging hard. Use -g to help debugging
CFLAGS   = -Wall -c $(DEBUG) 
LFLAGS   = -Wall -larmadillo $(DEBUG) 
INCLUDE  = -I$(PWD) -I$(PWD)/../ -I$(PWD)/../test # armadillo header files are in /usr/local/include, symlinked to the homebrew Cellar
LFLAGS_STATIC = -Wall -framework Accelerate $(DEBUG)

BLAS_LIB    = $(MATLAB_DIR)/bin/$(MATLAB_ARCH)/libmwblas.dylib
LAPACK_LIB  = $(MATLAB_DIR)/bin/$(MATLAB_ARCH)/libmwlapack.dylib

MEX         = $(MATLAB_DIR)/bin/mex
MEX_FLAGS   = -v -largeArrayDims
MEX_INCLUDE = -I./ -I/usr/local/include 
MEX_LFLAGS  = -Wall $(DEBUG) -L$(ARMADILLO_DIR) $(ARMADILLO_LIB)
MEX_CXXOPT  = CXXOPTIMFLAGS="$(OPT) $(DEBUG)"

ARMA_STATIC = -DARMA_DONT_USE_WRAPPER # to force linking to lapack and blas, with simple inclusion of header files
ARMA_FLAGS  = -DARMA_BLAS_LONG_LONG -DARMA_64BIT_WORD -DARMA_USE_CXX11 # These flags are important to allow armadillo to sync with Matlab

MEX_LFLAGS_STATIC = LINKLIBS='-L$(MATLAB_DIR)/bin/$(MATLAB_ARCH)/ -lmx -lmex -lm -lmat -framework Accelerate'  # LINKLIBS='-L$(MATLAB_DIR)/bin/$(MATLAB_ARCH)/ -lmx -lmex -lmat $(BLAS_LIB) $(LAPACK_LIB)' # - use these to link against matlab intel MKL libraries, not to MacOSX framework Accelerate #

# This line is very machine specific. Try commenting this out, and running mex in verbose mode - it should print out the defaults for your system. Unfortunately, setting this file up to just append to this variable seems impossible.
MEX_CXXFLAG = CXXFLAGS="-fno-common -arch x86_64 -march=native -mmacosx-version-min=10.9 -fexceptions -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX$(MACOS_VERSION).sdk -fobjc-arc -std=c++14 -stdlib=libc++ -Wall"





# What to do on calling make with no arguments
default: all


# Multiple target calls
all: mex test docs tidy


# Executable targets
test: $(OBJS)
	$(CC) $(OBJS) -o ../test/test $(INCLUDE) $(LFLAGS_STATIC) $(ARMA_FLAGS)

mex: $(TARGET).o
	$(MEX) ../$(TARGET)_mx.cpp $(TARGET).o $(MEX_FLAGS) $(MEX_CXXOPT) $(MEX_INCLUDE) $(MEX_CXXFLAG) $(ARMA_STATIC) $(MEX_LFLAGS_STATIC) $(ARMA_FLAGS)

# This also works, with no .o file:
#	/Applications/MATLAB_R2016b.app/bin/mex ./$(TARGET)_mx.cpp $(TARGET).cpp -v -largeArrayDims CXXOPTIMFLAGS="-O0"  -I/usr/local/include/ -I./  CXXFLAGS="-fno-common -arch x86_64 -march=native -mmacosx-version-min=10.9 -fexceptions -isysroot /Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX10.9.sdk -fobjc-arc -std=c++14 -stdlib=libc++" -DARMA_BLAS_LONG_LONG -I/usr/local/opt/armadillo/lib /usr/local/opt/armadillo/lib/libarmadillo.6.40.3.dylib
# -DARMA_NO_DEBUG CXXOPTIMFLAGS="-O3 -DNDEBUG"
# CXXOPTIMFLAGS="-O0"
# -DARMA_64BIT_WORD -DARMA_NO_WRAPPER


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
