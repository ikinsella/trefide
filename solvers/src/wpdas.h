#ifndef WPDAS_H
#define WPDAS_H

/* main routine for pdas l1tf solver */
int weighted_pdas(const int n,
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

#endif /* WPDAS_H */
