#ifndef UTILS_H
#define UTILS_H

//#ifdef  __cplusplus
//extern "C" {
//#endif

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

//#ifdef  __cplusplus
//}
//#endif
#endif /* UTILS_H */
