#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "welch.h"

/*  process command-line arguments and return problem data:
 *      y: time-series data, n: size of y  
 */
void process_args(int argc, char *argv[], size_t *pn, double **py)
{
    FILE   *fp;
    char   *ifile_y;
    size_t n;
    int    buf_size;
    double val;
    double *buf_start, *buf_end, *buf_pos;

    /* handle command line arguments */
    switch (argc)
    {
    case 2:
        ifile_y = argv[1];
        break;
    case 1:
        ifile_y = argv[0];
        break;
    default:
        fprintf(stdout,"usage: welch in_file [> out_file]\n");
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
    *py          = buf_start;
    *pn          = n;
}

/* Print vector of double type */
void print_dvec(const size_t n, const double *x)
{
    int i;
    fprintf(stdout,"\n");
    for (i = 0; i < n; i++)
        fprintf(stdout,"%e\n",x[i]);
    fprintf(stdout,"\n");
}

int main(int argc, char *argv[])
{
    size_t n;
    //size_t P;
    double *x;
    //double *psd;
    double var;

    /* process command line args to read time series y from file */
    process_args(argc, argv, &n, &x);

    /* call noise estimator */
    var = psd_noise_estimate(n, x);
    //P = (256/2) + 1;
    //psd = welch(n, 256, 128, 1, x); 
    /* print the result to stdout */
    fprintf(stdout, "Estimated Noise Variance: %1.2e", var);
    //print_dvec(P, psd);

    /* free allocated memory & exit */
    free(x);
    //free(psd);
    return(EXIT_SUCCESS);
}
