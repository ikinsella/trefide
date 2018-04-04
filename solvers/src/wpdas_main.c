#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "wpdas.h"

/*  process command-line arguments and return problem data:
 *      y: time-series data, n: size of y and lambda: regularization parameter
 */
void process_args(int argc, char *argv[], int *pn, double **py,
                  double *plambda)
{
    FILE   *fp;
    char   *ifile_y;
    int    c, n, buf_size;
    double lambda, val;
    double *buf_start, *buf_end, *buf_pos;
    int    rflag = 0;

    /* handle command line arguments */
    while ((c = getopt(argc, argv, "r")) != -1)
    {
        switch (c)
        {
        case 'r':
            rflag = 1;
            break;
        default:
            abort();
        }
    }
    argc -= optind;
    argv += optind;

    switch (argc)
    {
    case 2:
        lambda  = atof(argv[1]);
        ifile_y = argv[0];
        break;
    case 1:
        lambda  = 0.1;
        ifile_y = argv[0];
        break;
    default:
        fprintf(stdout,"usage: l1tf [-r] in_file [lambda] [> out_file]\n");
        exit (EXIT_SUCCESS);
    }

    /* read input data file */
    if ((fp = fopen(ifile_y,"r")) == NULL)
    {
        fprintf(stderr,"ERROR: Could not open file: %s\n",ifile_y);
        exit(EXIT_FAILURE);
    }

    /* store data in the buffer */
    buf_size    = 4096;
    buf_start   = malloc(sizeof(double)*buf_size);
    buf_end     = buf_start+buf_size;
    buf_pos     = buf_start;

    n = 0;
    while ( fscanf(fp,"%lg\n",&val) != EOF )
    {
        n++;
        *buf_pos++ = val;
        if (buf_pos >= buf_end) /* increase buffer when needed */
        {
	  buf_start = realloc(buf_start,sizeof(double)*buf_size*2);
            if (buf_start == NULL) exit(EXIT_FAILURE);
            buf_pos     = buf_start+buf_size;
            buf_size    *= 2;
            buf_end     = buf_start+buf_size;
        }
    }
    fclose(fp);

    /* set return values */
    *plambda     = lambda;
    *py          = buf_start;
    *pn          = n;
}

void print_info(const int n, const double lambda)
{

    fprintf(stderr,"--------------------------------------------\n");
    fprintf(stderr,"First order l1 trend filtering problem      \n");
    fprintf(stderr,"Solved via primal-dual active set algorithm \n");
    fprintf(stderr,"Written by: Ian Kinsella   Mar 28 2018      \n");
    fprintf(stderr,"--------------------------------------------\n");
    fprintf(stderr,"data length         = %d\n",n);
    fprintf(stderr,"lambda              = %e\n\n",lambda);
}

int main(int argc, char* argv[]){

    /* Initialize Variables */
    int n, iter;
    double *wi, *x, *y, *z;
    double lambda;

    /* process commendline arguments and read time series y from file */
    process_args(argc, argv, &n, &y, &lambda);

    /* print problem information */
    print_info(n, lambda);

    /* allocate memory for optimization variables */
    wi = malloc(sizeof(double)*n); // observation weights
    x = malloc(sizeof(double)*n); // primal variable
    z = malloc(sizeof(double)*(n-2)); // dual variable
    int i;    
    for (i = 0; i < n-2; i++) {
      z[i] = 0;
    }
    for (i = 0; i < n; i++) {
      wi[i] = 5;
    }
    
    /* Default arguments TODO: take as input */
    double p = 1; 
    int m = 5;
    double delta_s = 0.9; 
    double delta_e = 1.1;
    int maxiter = 500;
    int verbose = 1;
    
    /* call main solver */
    weighted_pdas(n, y, wi, lambda, x, z, &iter, p, m, delta_s, delta_e, maxiter, verbose);

    /* release allocated memory */
    free(wi);
    free(x);
    free(y);
    free(z);
    
    /* report exit status */
    return(EXIT_SUCCESS);
}
