#include <vector>
#include "wpdas.h"
#include "line_search.h"

#define STEP_PARTITION 60;

/******************************************************************************
 ***************************** Constrained Solver *****************************
 ******************************************************************************/

short constrained_wpdas(const int n,             // data length
                        const double *y,         // observations
                        const double *wi,        // inverse observation weights
                        const double delta,      // MSE constraint
                        double *x,               // primal variable
                        double *z,               // initial dual variable
                        double *lambda,          // initial regularization parameter
                        int *iters,            // pointer to iter # (so we can return it)
                        const int max_interp,  // number of times to try interpolating
                        const double tol,   // max num outer loop iterations
                        const int verbose)
{
    /* Declare & Initialize Local Variables */
    short status;
    double scale, tau;

    /* Compute Step Size wrt Transformed Lambda Space*/
    scale = compute_scale(n, y, delta);
    tau = (log(20 + (1 / scale)) - log(3 + (1 / scale))) / 60;

    /* If Uninitialized Compute Starting Point For Search */
    if (*lambda <= 0){
        *lambda = exp((log(20+(1/scale)) - log(3+(1/scale))) / 2 + log(3*scale + 1)) - 1;
    }

    /* v_k <- argmin_{v_k} ||v_k||_TF s.t. ||v - v_k||_2^2 <= T * delta */
    status = line_search(n, y, wi, delta, tau, x, z, lambda,
                         iters, max_interp, tol, verbose);

    return status;
}


short cps_wpdas(const int n,             // data length
                const double *y,         // observations
                const double *wi,        // inverse observation weights
                const double delta,      // MSE constraint
                double *x,               // primal variable
                double *z,               // initial dual variable
                double *lambda,          // initial regularization parameter
                int *iters,            // pointer to iter # (so we can return it)
                const double tol,   // max num outer loop iterations
                const int verbose)
{
    /* Declare & Initialize Local Variables */
    short status;
    double scale;

    /* If Uninitialized Compute Starting Point For Search */
    if (*lambda <= 0){
        scale = compute_scale(n, y, delta);
        *lambda = exp((log(20+(1/scale)) - log(3+(1/scale))) / 2 + log(3*scale + 1)) - 1;
    }

    /* v_k <- argmin_{v_k} ||v_k||_TF s.t. ||v - v_k||_2^2 <= T * delta */
    status = cps_tf(n, y, wi, delta, x, z, lambda, iters, tol, verbose);

    return status;
}


short cps_tf(const int n,           // data length
             const double *y,       // observations
             const double *wi,      // inverse observation weights
             const double delta,    // MSE constraint
             double *x,             // primal variable
             double *z,             // initial dual variable
             double *lambda,        // initial regularization parameter
             int *iters,            // pointer to iter # (so we can return it)
             const double tol,      // max num outer loop iterations
             const int verbose)
{
    double target = sqrt(n * delta);  // target norm of error
    short status;
    int iter;
    double l2_err;
    double l2_err_prev = 0;
    std::vector<double> resid(n, 0.0);

    /* pdas defaults */
    double p = 1;
    int m = 5;
    double delta_s = 0.9;
    double delta_e = 1.1;
    int maxiter = 500;
    int iters_local = *iters;

    while (iters_local < maxiter * 100)
    {
        /* Evaluate Solution At Lambda */
        status = weighted_pdas(n, y, wi, *lambda, x, z, &iter, p, m,
                               delta_s, delta_e, maxiter, verbose);
        if (unlikely(status < 0)){
            // TODO switch solvers
            return status;  // error within lagrangian solver
        }
        iters_local += iter;

        /* Compute norm of residual */
        vdSub(n, y, x, &resid[0]);
        l2_err = cblas_dnrm2(n, &resid[0], 1);

        /* Check For Convergence */
        if (fabs(target*target - l2_err*l2_err) / (target*target) < tol) {
            *iters += iters_local;
            return 1; // successfully converged within tolerance
        } else if(fabs(l2_err - l2_err_prev) < 1e-3) {
            *iters += iters_local;
            return 0;  // line search stalled
        }

        l2_err_prev = l2_err;

        /* Increment Lambda */
        // *lambda = exp(log(*lambda) + log(target) - log(l2_err));
        *lambda = (*lambda) * target / l2_err;
    }

    *iters += iters_local;
    return 0; // Line Search Didn't Converge
}

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
                  const int verbose)
{
  /************************** Initialize Variables ***************************/
    /* Line search internals */
    int iter, interp;
    short direction, status;
    double mse, mse_prev, err, tau_interp;

    /* pdas defaults */
    double p = 1;
    int m = 5;
    double delta_s = 0.9;
    double delta_e = 1.1;
    int maxiter = 500;

    /************************* Main Search Algorithm *************************/

    /* Evaluate Initialed Lambda */
    status = weighted_pdas(n, y, wi, *lambda, x, z, &iter, p, m,
                           delta_s, delta_e, maxiter, verbose);
    if (unlikely(status < 0)){
        // TODO switch solvers
        return status;  // error within lagrangian solver
    }
    *iters += iter;
    evaluate_search_point(n, y, wi, delta, x, &mse, &err);
    direction = sign(delta - mse);

    /************************** Interpolation Phase **************************/

    /* Perform K interpolations before transitioning to fixed step */
    for (interp = 0; interp < max_interp; interp++){

        /* Check to see if we landed in tol band */
        if (err <= tol){
            return 1; // successful solve
        }

        /* Take a step towards delta in transformed lambda space */
        *lambda = exp(log(*lambda + 1) + direction * tau) - 1;
        status = weighted_pdas(n, y, wi, *lambda, x, z, &iter, p, m,
                               delta_s, delta_e, maxiter, verbose);
        if (unlikely(status < 0)){
            // TODO switch solvers
            return status;  // error within lagrangian solver
        }
        *iters += iter;
        mse_prev = mse;
        evaluate_search_point(n, y, wi, delta, x, &mse, &err);

        /* Interpolate to next search point in transformed lambda space */
        tau_interp = tau * direction * (delta - mse) / (mse - mse_prev);
        *lambda = exp(log(*lambda + 1) + tau_interp) - 1;
        status = weighted_pdas(n, y, wi, *lambda, x, z, &iter, p, m,
                               delta_s, delta_e, maxiter, verbose);

        if (unlikely(status < 0)){
            // TODO switch solvers
            return status;  // error within lagrangian solver
        }
        *iters += iter;
        evaluate_search_point(n, y, wi, delta, x, &mse, &err);
        direction = sign(delta - mse);

    }

    /************************** Stepping Phase **************************/

    /* Step towards delta until we cross or land in tol band */
    while ((direction * sign(delta - mse) > 0) && (err > tol)){

        /* Take a step towards delta in transformed lambda space */
        *lambda = exp(log(*lambda + 1) + direction * tau) - 1;
        status = weighted_pdas(n, y, wi, *lambda, x, z, &iter, p, m,
                               delta_s, delta_e, maxiter, verbose);

        if (unlikely(status < 0)) {
            // TODO switch solvers
            return status;  // error within lagrangian solver
        }
        *iters += iter;
        mse_prev = mse;
        evaluate_search_point(n, y, wi, delta, x, &mse, &err);
        if (fabs(mse - mse_prev) < 1e-3){
            return 0; // Line search stalled, but wpdas didn't blow up
        }

    }

    /************************** Refine Estimate **************************/

    /* Interpolate to delta in transformed lambda space to refine estimate */
    // TODO ensure that interp leaves you between last two search points
    tau_interp = tau * direction * (delta - mse) / (mse - mse_prev);
    *lambda = exp(log(*lambda + 1) + tau_interp) - 1;
    status = weighted_pdas(n, y, wi, *lambda, x, z, &iter, p, m,
                           delta_s, delta_e, maxiter, verbose);

    if (unlikely(status < 0)){
        // TODO switch solvers
        return status;  // error within lagrangian solver
    }
    *iters += iter;
    return 1; // Successful Solve
}
