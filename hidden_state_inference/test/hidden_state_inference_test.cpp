/*! \file hidden_state_inference_test.cpp 
 *
 *  \brief Test working of hidden_state_inference.cpp
 *
 * Compile and run to test hidden_state_inference.cpp
 *
 * Giles Colclough
 * 2 Feb 2016
 *
 *
 *	Copyright 2015 OHBA
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


#include "hidden_state_inference_test.hpp"
  
//! Run tests on hidden_state_inference
/*!  
 * Performs two runs of hidden state inference, to check interface and performance. 
 */
int main(int argc, char* const argv[]) {
    
    /* Problem dimensions */
    int nSamples {100};
    int nClasses {3};
	int order    {0};
    
    arma::arma_rng::set_seed(0);
    
    /* Declare memory for answers */
    arma::mat    gamma(nSamples-order, nClasses);
    arma::mat    Xi(nSamples-1-order, std::pow(nClasses,2));
	arma::colvec scale(nSamples);
    
    /* Set up problem */
    arma::mat B(nSamples, nClasses, arma::fill::randu); // element-wise probabilities
    arma::rowvec Pi_0 {0.2, 0.4, 0.4};
    arma::mat A {{0.8, 0.1, 0.1},
                 {0.3, 0.5, 0.2}, 
                 {0.1, 0.4, 0.5}};
                 
    /* Solve! */
    hidden_state_inference(gamma, Xi, scale, B, Pi_0, A, order);
    
    gamma.print();
    Xi.print();
	
	/* clear before running again */
	gamma.fill(0.0);
	Xi.fill(0.0);
    
    /* Try again with fixed state movement */
    arma::rowvec Pi_1 {0.9, 0.1, 0.0};
    arma::mat    A_1 {{1.0, 0.0, 0.0},
                      {0.0, 1.0, 0.0}, 
                      {0.0, 0.0, 1.0}};
                 
    hidden_state_inference(gamma, Xi, scale, B, Pi_1, A_1);
    
    gamma.print();
    Xi.print();
    return 0;
}