// hidden_state_inference_mx.hpp
// 21-01-2016 Giles Colclough

#ifndef __HIDDEN_STATE_INFERENCE_MX__
#define __HIDDEN_STATE_INFERENCE_MX__

#include "armadillo" /* statically linked to avoid errors in matlab */
#include <cmath>
#include "mex.h"
#include "hidden_state_inference.hpp"


/* function declarations */
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]);

#endif
/* EOF */