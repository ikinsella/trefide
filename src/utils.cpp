#include "utils.h"
/******************************************************************************
 *                            Utility Functions                               *
 ******************************************************************************/

/* Print vector of double type */
void print_dvec(const int n, const double *x) {
    std::cout << "\n";
    for (int i = 0; i < n; i++)
        std::cout << x[i] << std::endl;
    std::cout << "\n";
}
