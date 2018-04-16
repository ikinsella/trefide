#ifndef PDAS_SG2_H
#define PDAS_SG2_H

/* main routine for l1 trend filtering */
int active_set(const int n,
	       const double *y,
	       const double lambda,
	       double *x,
	       double *z,
	       double p,
	       const int m,
	       const double delta_s,
	       const double delta_e,
	       const int maxiter,
	       const int verbose);

/* description */
void   Dx(const int n,
	  const double *x,
	  double *y);

/* another desciption */
void   DTx(const int n,
	   const double *x,
	   double *y);

/* yet another description 8? */
void print_dvec(const int n,
		const double *x);

#endif /* PDAS_SG2_H */
