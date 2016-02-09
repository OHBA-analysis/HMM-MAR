/* test_armadillo
 *
 * Giles Colclough
 *
 * Tests armadillo library on unix. 
 */

#include <armadillo>

int main() {
    arma::mat A(5,5, arma::fill::randu);
    A.print("A:");
    return 0;
}
