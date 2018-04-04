#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>


/******************************************************************************
 *                             Matrix Operators                               *
 ******************************************************************************/


/* Computes y = D*x, where x has length n
 *
 *     | -1  2  -1  0  0 |
 * y = | 0  -1  2  -1  0 |*x
 *     | 0  0  -1  2  -1 |
 */
void Dx(const int n, const double *x, double *y)
{
    int i;
    for (i = 0; i < n-2; i++, x++)
        *y++ = -*x + *(x+1) + *(x+1) - *(x+2); /* y[0..n-3]*/
}


/* Computes y = D^T*x, where x has length n
 *
 *     | -1   0   0 |
 *     |  2  -1   0 |
 * y = | -1   2  -1 |*x
 *     |  0  -1   2 |
 *     |  0   0  -1 |
 */
void DTx(const int n, const double *x, double *y)
{
    int i;
    *y++ = -*x;                          /* y[0]     */
    *y++ = *x+*x-*(x+1);                 /* y[1]     */
    for (i = 2; i < n; i++,x++)
        *y++ = -*x+*(x+1)+*(x+1)-*(x+2); /* y[2..n-1]*/
    *y++ = -*x+*(x+1)+*(x+1); x++;       /* y[n]     */
    *y = -*x;                            /* y[n+1]   */
}


/******************************************************************************
 *                            Utility Functions                               *
 ******************************************************************************/


/* Print vector of double type */
void print_dvec(const int n, const double *x)
{
    int i;
    fprintf(stdout,"\n");
    for (i = 0; i < n; i++)
        fprintf(stdout,"%e\n",x[i]);
    fprintf(stdout,"\n");
}
