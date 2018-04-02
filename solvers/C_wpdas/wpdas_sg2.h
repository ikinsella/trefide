#ifndef PDAS_SG2_H
#define PDAS_SG2_H

/* main routine for l1 trend filtering */
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

/* description */
void   Dx(const int n,
	  const double *x,
          const double *wi,
	  double *y);

/* another desciption */
void   DTx(const int n,
	   const double *x,
	   double *y);

/* yet another description 8? */
void print_dvec(const int n,
		const double *x);

#endif /* PDAS_SG2_H */
