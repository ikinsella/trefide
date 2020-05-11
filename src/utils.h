#ifndef UTILS_H
#define UTILS_H

#include <iostream>

/******************************************************************************
 *                             Matrix Operators                               *
 ******************************************************************************/

/* Computes y = D*x, where x has length n
 *
 *     | -1  2  -1  0  0 |
 * y = | 0  -1  2  -1  0 |*x
 *     | 0  0  -1  2  -1 |
 */
inline void Dx(const int n, const double* x, double* y)
{
    for (int i = 0; i < n - 2; i++)
        y[i] = -x[i] + x[i + 1] + x[i + 1] - x[i + 2]; /* y[0..n-3]*/
}

/* Computes y = D^T*x, where x has length n
 *
 *     | -1   0   0 |
 *     |  2  -1   0 |
 * y = | -1   2  -1 |*x
 *     |  0  -1   2 |
 *     |  0   0  -1 |
 */
inline void DTx(const int n, const double* x, double* y)
{
    *y++ = -*x; /* y[0]     */
    *y++ = *x + *x - *(x + 1); /* y[1]     */
    for (int i = 2; i < n; i++, x++)
        *y++ = -*x + *(x + 1) + *(x + 1) - *(x + 2); /* y[2..n-1]*/
    *y++ = -*x + *(x + 1) + *(x + 1);
    x++; /* y[n]     */
    *y = -*x; /* y[n+1]   */
}

/* Utility Functions */

/* Print vector of double type */
void print_dvec(const int n, const double* x);

#endif /* UTILS_H */
