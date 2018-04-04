#ifndef UTILS_H
#define UTILS_H

/* y = D*x */
void   Dx(const int n,
	  const double *x,
	  double *y); 

/* y = D'*x */
void   DTx(const int n,
	   const double *x,
	   double *y);

/* yet another description */
void print_dvec(const int n,
		const double *x);

#endif /* UTILS_H */
