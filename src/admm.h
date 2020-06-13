#ifndef ADMM_H
#define ADMM_H

#include "../external/glmgen/include/cs.h"

/**
 * ...
 *
 * @param n data length
 * @param x data locations
 * @param y data observations
 * @param w data observation weights
 * @param delta MSE constraint (noise var estimate)
 * @param beta primal variable
 * @param alpha
 * @param u
 * @param lambda initial regularization parameter
 * @param iters pointer to iter # (so we can return it)
 * @param tol relative difference from target MSE
 * @param verbose
 * @return status
 */
short constrained_tf_admm(const int n, double *x, double *y, double *w,
                          const double delta, double *beta, double *alpha,
                          double *u, double *lambda, int *iters,
                          const double tol = 5e-2, const int verbose = 0);

/**
 * ...
 *
 * @param n data length
 * @param degree
 * @param x data locations
 * @param y data observations
 * @param w data observation weights
 * @param DktDk Difference Gram
 * @param delta MSE constraint (noise var estimate)
 * @param beta primal variable
 * @param alpha
 * @param u
 * @param lambda initial regularization parameter
 * @param rho
 * @param iters pointer to iter # (so we can return it)
 * @param tol relative difference from target MSE
 * @param verbose
 * @return -1 aborted search, 1 successfully converged, 0 tol not reached
 */
short cps_tf_admm(const int n, const int degree, double *x, double *y,
                  double *w, cs *DktDk, const double delta, double *beta,
                  double *alpha, double *u, double *lambda, double rho,
                  int *iters, const double tol, const int verbose);

/**
 * ...
 *
 * @param n data length
 * @param x data locations
 * @param y data observations
 * @param w data observation weights
 * @param lambda regularization parameter
 * @param beta primal variable
 * @param alpha
 * @param u
 * @param iter pointer to iter # (so we can return it)
 * @param verbose
 * @return status
 */
short langrangian_tf_admm(const int n, double *x, double *y, double *w,
                          double lambda, double *beta, double *alpha, double *u,
                          int *iter, const int verbose = 0);

#endif
