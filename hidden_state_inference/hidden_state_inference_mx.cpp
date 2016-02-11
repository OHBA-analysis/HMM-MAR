/*! \file hidden_state_inference_mx.cpp
 *  \brief Mex interface for hidden state inference in VB-HMM
 *
 * Hidden state inference for VB-solved HMM, using the forwards-backwards 
 * algorithm.
 *
 *
 *
 * To use this file, you may need to recompile, changing the top of the makefile to reflect your matlab installation
 *
 *
 *	Copyright 2016 OHBA
 *
 *	This program is free software: you can redistribute it and/or modify
 *	it under the terms of the GNU General Public License as published by
 *	the Free Software Foundation, either version 3 of the License, or
 *	(at your option) any later version.
 *
 *	This program is distributed in the hope that it will be useful,
 *	but WITHOUT ANY WARRANTY; without even the implied warranty of
 *	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *	GNU General Public License for more details.
 *
 *	You should have received a copy of the GNU General Public License
 *	along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *
 *	$LastChangedBy$
 *	$Revision$
 *	$LastChangedDate$
 *	Contact: giles.colclough@ohba.ox.ac.uk
 *	Originally written on: MACI64 by Giles Colclough, 14-Jul-2015
 */


#include "hidden_state_inference_mx.hpp"





//! Mex interface for hidden state inference using forward-backward algorithm
/*! mexFunction(int num_outputs, mxArray* outputs, int num_inputs, mxArray* inputs)
 *
 * HIDDEN_STATE_INFERENCE using foward-backward propagation
 *
 * [GAMMA,XI,SCALE] = HIDDEN_STATE_INFERENCE(B, PI_0, P, ORDER)
 *
 *   returns the probability of each state given the data, GAMMA, and the 
 *   probability of each state given the parents and children, XI, together 
 *   with the scalings on alpha, SCALE. 
 *
 *   The function uses the marginal likelihood of the data given the priors 
 *   for each state, the initial probabilities, PI_0, and the transition 
 *   probability matrix A. There is also an order term, relevant for MAR
 *   models. If you're not using a MAR model, set order = 0. 
 *
 *  You MUST have four inputs and three outputs. If you don't want to make
 *  / assign variables, use ~ as a dummy variable 
 *  e.g. [g,x,~] = hidden_state_inference_mx(B,PI,P,ORDER)
 */
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    /* check for proper number of arguments */
    if(nrhs!=4) {
        mexErrMsgIdAndTxt("hidden_state_inference_mx:Inputs","Four inputs required.");
    }
    if(nlhs!=3) {
        mexErrMsgIdAndTxt("hidden_state_inference_mx:Outputs","Three outputs required.");
    }
    
    /* assign inputs to variables */
    double* B_;        // input 1 - probabilities of each element
    double* PI_0_;     // input 2 - initial probabilities
    double* A_;        // input 3 - transition probabilities
	int     order_;    // input 4 - HMM order
    size_t  nrows;
    size_t  ncols;
    double* gamma_;    // output 1
    double* Xi_;       // output 2
	double* scale_;    // output 3
    
    B_     = (double*) mxGetPr(prhs[0]);
    PI_0_  = (double*) mxGetPr(prhs[1]);
    A_     = (double*) mxGetPr(prhs[2]);
	order_ = (int)     mxGetScalar(prhs[3]);
    
    /* get dimensions of the input matrix */
    nrows = mxGetM(prhs[0]);
    ncols = mxGetN(prhs[0]);
    int nSamples = (int) nrows; //{static_cast<int> (nrows)};
    int nClasses = (int) ncols; //{static_cast<int> (ncols)};  
    
    /* Check dimensions match */
    if (1 != mxGetM(prhs[1]) && mxGetN(prhs[1]) != ncols) { 
        mexErrMsgIdAndTxt("hidden_state_inference_mx:muDims", \
                          "mu should be a row vector as long as the number of columsn in X. \n");
    }
    if (mxGetM(prhs[2]) != ncols && mxGetN(prhs[2]) != ncols) {
        mexErrMsgIdAndTxt("hidden_state_inference_mx:muDims", \
                          "R should be square, of same size as columns of X. \n");
    }
    
    /* create the output matrix */
    plhs[0] = mxCreateDoubleMatrix((mwSize) nrows - order_, (mwSize) ncols, mxREAL);
    plhs[1] = mxCreateDoubleMatrix((mwSize) nrows - 1 - order_, (mwSize) std::pow(ncols, 2), mxREAL);
	plhs[2] = mxCreateDoubleMatrix((mwSize) nrows, (mwSize) 1, mxREAL);
    
    /* get a pointer to the real data in the output matrices */
    gamma_ = mxGetPr(plhs[0]);
    Xi_    = mxGetPr(plhs[1]);
    scale_ = mxGetPr(plhs[2]);
    
    /* assign input matrices to arma mat*/
    arma::mat    B (nSamples, nClasses);
    arma::rowvec Pi_0 (nClasses);
    arma::mat    A (nClasses, nClasses);
    arma::uword  ii = 0;
	
    for (int index = 0; index < nSamples * nClasses; index++) {
        ii    = index;
        B(ii) = B_[index];
    }
    for (int index = 0; index < nClasses; index++) {
        ii       = index;
        Pi_0(ii) = PI_0_[index];
    }
    for (int index = 0; index < std::pow(nClasses,2); index++) {
        ii    = index;
        A(ii) = A_[index];
    }
    
    /* Create output matrix */
    arma::mat    gamma(nSamples-order_, nClasses); 
    arma::mat    Xi(nSamples-1-order_, std::pow(nClasses,2));
    arma::colvec scale(nSamples);
    
    /* Run worker function */
    hidden_state_inference(gamma, Xi, scale, B, Pi_0, A, order_);
	
    /* Save out results */
    for (int index = 0; index < (nSamples - order_) * nClasses; index++) {
        ii            = index;
        gamma_[index] = gamma(ii);
    }
    for (int index = 0; index < (nSamples - 1 - order_) * std::pow(nClasses,2); index++) {
        ii         = index;
        Xi_[index] = Xi(index);
    }
	for (int index = 0; index < nSamples; index++) {
		scale_[index] = scale(index);
	}
    
    return;
}
/* [EOF] */
