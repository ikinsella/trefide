#ifndef LINE_SEARCH_H
#define LINE_SEARCH_H

//#ifdef  __cplusplus
//extern "C" {
//#endif

double compute_scale(const int t, 
                     const double *y, 
                     const double delta);

short line_search(const int n,           // data length
      		  const double *y,       // observations
		  const double *wi,      // inverse observation weights
                  const double delta,    // MSE constraint	
                  const double tau,      // step size in transformed space
                  double *x,             // primal variable
		  double *z,             // initial dual variable
		  double *lambda,        // initial regularization parameter
		  int *iters,            // pointer to iter # (so we can return it)
                  const int max_interp,  // number of times to try interpolating
		  const double tol,      // max num outer loop iterations
		  const int verbose);

short line_search(const int n,           // data length
      		  const double *y,       // observations
		  const double *wi,      // inverse observation weights
                  const double delta,    // MSE constraint	
                  double *x,             // primal variable
		  double *z,             // initial dual variable
		  double *lambda,        // initial regularization parameter
		  int *iters,            // pointer to iter # (so we can return it)
		  const double tol,      // max num outer loop iterations
		  const int verbose);

short constrained_wpdas(const int n,             // data length
                        const double *y,         // observations
                        const double *wi,        // inverse observation weights
                        const double delta,      // MSE constraint	
                        double *x,               // primal variable
                        double *z,               // initial dual variable
                        double *lambda,          // initial regularization parameter
                        int *iters,            // pointer to iter # (so we can return it)
                        const int max_interp=1,  // number of times to try interpolating
                        const double tol=1e-3,   // max num outer loop iterations
                        const int verbose=0);

short cps_tf(const int n,           // data length
             const double *y,       // observations
             const double *wi,      // inverse observation weights
             const double delta,    // MSE constraint	
             double *x,             // primal variable
             double *z,             // initial dual variable
             double *lambda,        // initial regularization parameter
             int *iters,            // pointer to iter # (so we can return it)
             const double tol,      // max num outer loop iterations
             const int verbose);

short cps_wpdas(const int n,             // data length
                const double *y,         // observations
                const double *wi,        // inverse observation weights
                const double delta,      // MSE constraint	
                double *x,               // primal variable
                double *z,               // initial dual variable
                double *lambda,          // initial regularization parameter
                int *iters,            // pointer to iter # (so we can return it)
                const double tol=1e-3,   // max num outer loop iterations
                const int verbose=0);
//#ifdef  __cplusplus
//}
//#endif
#endif /* LINE_SEARCH_H */
