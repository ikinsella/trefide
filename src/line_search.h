#ifndef LINE_SEARCH_H
#define LINE_SEARCH_H

#include <math.h>
#include <mkl.h>

# define FORCEINLINE __attribute__((always_inline)) inline
#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)


inline short sign(double val) {
    return (0 < val) - (val < 0);
}

/* Computes Scaling Factor For 2nd Order TF Line Search */
inline double compute_scale(const int t,
                     const double *y,
                     const double delta)
{
    /* Compute power of signal: under assuming detrended and centered */
    double var_y = cblas_dnrm2(t, y, 1);
    var_y *= var_y;
    var_y /= t;

    /* Return scaling factor sigma_eps / sqrt(SNR) */
    if (var_y <= delta)
        return sqrt(var_y) / sqrt(.1); // protect against faulty noise estimates

    return delta / sqrt(var_y - delta);
}


inline void evaluate_search_point(const int n,
                           const double *y,
                           const double *wi,
                           const double delta,
                           const double *x,
                           double *mse,
                           double *err)
{
    /* Compute weighted mse */
    double _mse = 0.0;

    for (int i = 0; i < n; i++)
        _mse += (y[i] - x[i]) * (y[i] - x[i]) / wi[i];

    _mse /= n;
    *mse = _mse;

    /* compute % error */
    *err = fabs(delta - _mse) / delta;
}

void evaluate_search_point(const int n,
                           const double *y,
                           const double *wi,
                           const double delta,
                           const double *x,
                           double *mse,
                           double *err);

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

#endif /* LINE_SEARCH_H */
