/*! \file hidden_state_inference.cpp 
 *
 *  \brief Hidden state inference for the HMM
 *
 * Hidden state inference for VB-solved HMM, using the forwards-backwards 
 * algorithm.
 *
 * This code is an implementation of Diego Vidaurre's nodecluster algorithm 
 * for C++. 
 *
 * Giles Colclough
 * 2 Feb 2016
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
 *	Originally written on: MACI64 by Giles Colclough, 02-Feb-2016
 */




#include "hidden_state_inference.hpp"



//! hidden state inference
/*!
 * void hidden_state_inference(gamma, Xi, B, Pi_0, P)
 *
 * A function to perform inference of hidden states in an HMM. 
 *
 *
 *
 * The outputs, gamma and Xi, need to be passed in with memory already declared. 
 *
 * gamma - (nSamples - order) x nClasses
 *
 * Xi    - (nSamples-1-order) x nClasses**2
 *
 * scale - (nSamples x 1)
 *
 *
 * The inputs are
 *
 * B    - nSamples x nClasses
 *
 * Pi_0 - 1 x nClasses
 *
 * P    - nClasses x nClasses
 *
 * order - integer
 * 
 */
void hidden_state_inference(arma::mat          &gamma,   //!<[in] Probability of hidden state given data
							arma::mat          &Xi,      //!<[in] Probability of hidden state given child and parent states, given data
							arma::colvec       &scale,   //!<[in] scaling factor of alpha.
							const arma::mat    &B,       //!<[in] Probability of individual data point, under observation model for each state
							const arma::rowvec &Pi_0,    //!<[in] Initial state probabilities
							const arma::mat    &P,       //!<[in] State transition probabilities
							const int          order     //!<[in] order of MAR model [default = 0 for MVN]
							) {
	
	/* define a very small number */
    static const double EPS = std::numeric_limits<double>::min();
    
	/* Memory declaration */
    arma::uword  nSamples = B.n_rows;
    arma::uword  nClasses = B.n_cols; 
    arma::mat    alpha(nSamples - order, nClasses);
    arma::mat    beta(nSamples - order, nClasses);
    arma::mat    t(nClasses, nClasses);
	
	/* check input dimensions */
	#ifndef NDEBUG
	    if (gamma.n_rows + order != nSamples || gamma.n_cols != nClasses) {
			throw std::runtime_error("Gamma should be (nSamples - order) x nClasses. ");
		}
		if (Xi.n_rows + order + 1 != nSamples || Xi.n_cols != std::pow(nClasses,2)) {
			throw std::runtime_error("Xi should be (nSamples - order - 1) x nClasses**2. ");
		}
		if (scale.n_elem != nSamples) {
			throw std::runtime_error("scale should have nSamples elements. ");
		}
	#endif
    
	/* fill in memory below order 
	 * 
	 * Diego keeps scale as full length, no matter the order. 
	 * This forces us to do some pretty weird vector filling below. 
	 * The logic is most readable if you set order=0 in your head. 
	 * We need to follow his conventions so that we can interface properly
	 * with the matlab code. 
	 */
	if (order > 0) {
		for (int i = 0; i < order; i++) {
			scale(i) = 0.0;
		}
	}
	
    /* forward pass */
    alpha.row(0)  = Pi_0 % B.row(order);
    scale(order)  = arma::sum(alpha.row(0));
    alpha.row(0) /= scale(order) + EPS;
    
    for (int i = 1 + order; i < (int) nSamples; i++) {
        alpha.row(i-order)  = (alpha.row(i-1-order) * P) % B.row(i);
        scale(i)            = arma::sum(alpha.row(i-order));
        alpha.row(i-order) /= scale(i) + EPS;
    }
    // scale.elem( find(scale < EPS) ) = EPS; //check for zeros
    
    /* backward pass */
    beta.row(nSamples - 1 - order) = arma::ones<arma::rowvec>(nClasses) / (scale(nSamples - 1) + EPS);
    for (int i = nSamples - 2; i>=order; i--) {
        beta.row(i-order)  = (beta.row(i+1-order) % B.row(i+1)) * P.t();
        beta.row(i-order) /= scale(i) + EPS;
        // check for infinities
	}
    
    /* marginal probabilities */
    gamma             = alpha % beta;
    gamma.each_col() /= arma::sum(gamma,1); // divide by row sums
    
    for (int i = order; i < (int) nSamples - 1; i++) {
        t                = P % ((alpha.row(i - order)).t() * (beta.row(i+1-order) % B.row(i+1)));
        Xi.row(i-order)  = reshape(t, 1, std::pow(nClasses,2));
        Xi.row(i-order) /= (accu(t) + EPS);
    }   
    return;
}
/* [EOF] */