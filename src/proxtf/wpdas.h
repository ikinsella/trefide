#ifndef WPDAS_H
#define WPDAS_H

//#ifdef  __cplusplus
//extern "C" {
//#endif

/* main routine for pdas l1tf solver */
short weighted_pdas(const int n,
                    const double *y,
                    const double *wi,	       
                    const double lambda,
                    double *x,
                    double *z,
                    int *iter,
                    double p,
                    const int m,
                    const double delta_s,
                    const double delta_e,
                    const int maxiter,
                    const int verbose);

//#ifdef  __cplusplus
//}
//#endif
#endif /* WPDAS_H */
