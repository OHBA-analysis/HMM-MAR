/* hidden_state_inference.hpp */

#ifndef __HIDDEN_STATE_INFERENCE__
#define __HIDDEN_STATE_INFERENCE__

#include "armadillo"
#include <cmath>
#include <stdexcept>

void hidden_state_inference(arma::mat          &gamma,   //!<[in] Probability of hidden state given data
							arma::mat          &Xi,      //!<[in] Probability of hidden state given child and parent states, given data
							arma::colvec       &scale,   //!<[in] scaling factor of alpha.
							const arma::mat    &B,       //!<[in] Probability of individual data point, under observation model for each state
							const arma::rowvec &Pi_0,    //!<[in] Initial state probabilities
							const arma::mat    &P,       //!<[in] State transition probabilities
							const int          order = 0 //!<[in] order of MAR model [default = 0 for MVN]
							);
#endif