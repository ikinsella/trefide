#include <stdlib.h>
#include <math.h>
#include <mkl.h>
#include "wpdas.h"

int STEP_PARTITION = 60;

/******************************************************************************
 **************************** Function Declarations ***************************
 ******************************************************************************/

short sign(double val);

void evaluate_search_point(const int n,
                           const double *y, 
                           const double *wi, 
                           const double delta, 
                           const double *x, 
                           double *mse, 
                           double *err);

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
                        const int max_interp=1,  // number of times to try interpolating
                        const double tol=1e-3,   // max num outer loop iterations
                        const int verbose=0)
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
                const double tol=1e-3,   // max num outer loop iterations
                const int verbose=0)
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
    /* Declar & allocate internal vars */
    double target = sqrt(n*delta);  // target norm of error
    short status;
    int iter;
    double l2_err, l2_err_prev = 0;
    double * resid = (double *) malloc(n * sizeof(double));

    /* pdas defaults */
    double p = 1; 
    int m = 5;
    double delta_s = 0.9; 
    double delta_e = 1.1;
    int maxiter = 500;

    while (*iters < maxiter * 100)
    {
        /* Evaluate Solution At Lambda */
        status = weighted_pdas(n, y, wi, *lambda, x, z, &iter, p, m, 
                               delta_s, delta_e, maxiter, verbose);
        if (status < 0){
            free(resid);
            // TODO switch solvers
            return status;  // error within lagrangian solver
        }
        *iters += iter;

        /* Compute norm of residual */
        vdSub(n, y, x, resid);
        l2_err = cblas_dnrm2(n, resid, 1);

        /* Check For Convergence */
        if (fabs(target*target - l2_err*l2_err) / (target*target) < tol){
            free(resid);
            return 1; // successfully converged within tolerance
        } else if(fabs(l2_err - l2_err_prev) < 1e-3){
            free(resid);
            return 0;  // line search stalled
        } 
        l2_err_prev = l2_err;

        /* Increment Lambda */
        *lambda = exp(log(*lambda) + log(target) - log(l2_err));
    }
    free(resid);
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
    if (status < 0){
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
        if (status < 0){
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
        if (status < 0){
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
        if (status < 0){
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
    if (status < 0){
        // TODO switch solvers
        return status;  // error within lagrangian solver
    }
    *iters += iter;
    return 1; // Successful Solve 
}


/******************************************************************************
 ***************************** Utility Functions ******************************
 ******************************************************************************/


/* Computes Scaling Factor For 2nd Order TF Line Search */
double compute_scale(const int t, 
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


void evaluate_search_point(const int n, 
                           const double *y, 
                           const double *wi, 
                           const double delta, 
                           const double *x, 
                           double *mse, 
                           double *err)
{
    int i; 

    /* Compute weighted mse */
    *mse = 0;
    for (i = 0; i < n; i++, x++, y++, wi++)
        *mse += pow(*y - *x, 2) / *wi;
    *mse /= n;
        
    /* compute % error */
    *err = fabs(delta - *mse) / delta;

}


short sign(double val) {
    return (0 < val) - (val < 0);
}
