#ifndef ADMM_H
#define ADMM_H

# include "../glmgen/include/cs.h"
short cps_tf_admm(const int n,        // data length
                  const int degree,
                  double* x,          // data locations
                  double *y,          // data observations
                  double *w,          // data observation weights
                  cs * DktDk,         // Difference Gram
                  const double delta, // MSE constraint (noise var estimate)	
                  double *beta,       // primal variable
                  double *alpha,
                  double *u,
                  double *lambda,     // initial regularization parameter
                  double rho,
                  int *iters,         // pointer to iter # (so we can return it)
                  const double tol,   // relative difference from target MSE
                  const int verbose);


short constrained_tf_admm(const int n,           // data length
                          double* x,             // data locations
                          double *y,             // data observations
                          double *w,             // data observation weights
                          const double delta,    // MSE constraint (noise var estimate)	
                          double *beta,          // primal variable
                          double *alpha,
                          double *u,
                          double *lambda,        // initial regularization parameter
                          int *iters,            // pointer to iter # (so we can return it)
                          const double tol=5e-2, // relative difference from target MSE
                          const int verbose=0);


short langrangian_tf_admm(const int n,           // data length
                          double* x,             // data locations
                          double *y,             // data observations
                          double *w,             // data observation weights
                          double lambda,        // regularization parameter
                          double *beta,          // primal variable
                          double *alpha,
                          double *u,
                          int *iter,             // pointer to iter # (so we can return it)
                          const int verbose=0);

#endif
