#include "utils.h"
/******************************************************************************
 *                            Utility Functions                               *
 ******************************************************************************/


/* Print vector of double type */
void print_dvec(const int n, const double *x)
{
    fprintf(stdout,"\n");
    for (int i = 0; i < n; i++)
        fprintf(stdout,"%e\n",x[i]);
    fprintf(stdout,"\n");
}
