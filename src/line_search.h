#ifndef LINE_SEARCH_H
#define LINE_SEARCH_H

#include <math.h> /* sqrt, fabs */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
#include <mkl.h> /* cblas_dnrm2 */
#pragma GCC diagnostic pop

inline int check_sign(double val) { return (0 < val) - (val < 0); }

/* Computes Scaling Factor For 2nd Order TF Line Search */
inline double compute_scale(const int t, const double *y, const double delta) {
    /* Compute power of signal: under assuming detrended and centered */
    double var_y = cblas_dnrm2(t, y, 1);
    var_y *= var_y;
    var_y /= t;

    /* Return scaling factor sigma_eps / sqrt(SNR) */
    if (var_y <= delta)
        return sqrt(var_y) / sqrt(.1); // protect against faulty noise estimates

    return delta / sqrt(var_y - delta);
}

inline void evaluate_search_point(const int n, const double *y,
                                  const double *wi, const double delta,
                                  const double *x, double *mse, double *err) {
    /* Compute weighted mse */
    double _mse = 0.0;

    for (int i = 0; i < n; i++)
        _mse += (y[i] - x[i]) * (y[i] - x[i]) / wi[i];

    _mse /= n;
    *mse = _mse;

    /* compute % error */
    *err = fabs(delta - _mse) / delta;
}

/**
 * ...
 *
 * @param n Data length
 * @param y Observations
 * @param wi Inverse observation weights
 * @param delta MSE constraint
 * @param tau Step size in transformed space
 * @param x Primal variable
 * @param z Initial dual variable
 * @param lambda Initial regularization parameter
 * @param iters Pointer to iter # (so we can return it)
 * @param max_interp Number of times to try interpolating
 * @param tol Max number of outer loop iterations
 * @param verbose
 * @return status
 */
int line_search(const int n, const double *y, const double *wi,
                const double delta, const double tau, double *x, double *z,
                double *lambda, int *iters, const int max_interp,
                const double tol, const int verbose);

/**
 * ...
 *
 * @param n Data length
 * @param y Observations
 * @param wi Inverse observation weights
 * @param delta MSE constraint
 * @param x Primal variable
 * @param z Initial dual variable
 * @param lambda Initial regularization parameter
 * @param iters Pointer to iter # (so we can return it)
 * @param max_interp Number of times to try interpolating
 * @param tol Max number of outer loop iterations
 * @param verbose
 * @return status
 */
int constrained_wpdas(const int n, const double *y, const double *wi,
                      const double delta, double *x, double *z, double *lambda,
                      int *iters, const int max_interp = 1,
                      const double tol = 1e-3, const int verbose = 0);

/**
 * ...
 *
 * @param n Data length
 * @param y Observations
 * @param wi Inverse observation weights
 * @param delta MSE constraint
 * @param x Primal variable
 * @param z Initial dual variable
 * @param lambda Initial regularization parameter
 * @param iters Pointer to iter # (so we can return it)
 * @param tol Max number of outer loop iterations
 * @param verbose
 * @return status
 */
int cps_tf(const int n, const double *y, const double *wi, const double delta,
           double *x, double *z, double *lambda, int *iters, const double tol,
           const int verbose);

/**
 * ...
 *
 * @param n Data length
 * @param y Observations
 * @param wi Inverse observation weights
 * @param delta MSE constraint
 * @param x Primal variable
 * @param z Initial dual variable
 * @param lambda Initial regularization parameter
 * @param iters Pointer to iter # (so we can return it)
 * @param tol Max number of outer loop iterations
 * @param verbose
 * @return status
 */
int cps_wpdas(const int n, const double *y, const double *wi,
              const double delta, double *x, double *z, double *lambda,
              int *iters, const double tol = 1e-3, const int verbose = 0);

#endif /* LINE_SEARCH_H */
