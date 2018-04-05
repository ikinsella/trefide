#include <stdlib.h>
#include <math.h>
#include "wpdas.h"


int sign(double val) {
    return (0 < val) - (val < 0);
}

void evaluate_search_point(const int n,
                           const double *y, 
                           const double *wi, 
                           const double delta, 
                           const double *x, 
                           double *mse, 
                           double *err);

int line_search(const int n,           // data length
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
    int iter, interp, direction;
    double mse, mse_prev, err, tau_interp;
    
    /* pdas defaults */
    double p = 1; 
    int m = 5;
    double delta_s = 0.9; 
    double delta_e = 1.1;
    int maxiter = 500;
    
    /************************* Main Search Algorithm *************************/

    /* Evaluate Initialed Lambda */
    weighted_pdas(n, y, wi, *lambda, x, z, &iter, p, m, delta_s, delta_e, maxiter, verbose);
    *iters += iter;
    evaluate_search_point(n, y, wi, delta, x, &mse, &err);
    direction = sign(delta - mse);

    /************************** Interpolation Phase **************************/
    
    /* Perform K interpolations before transitioning to fixed step */
    for (interp = 0; interp < max_interp; interp++){
    
        /* Check to see if we landed in tol band */
        if (err <= tol){
            return 1;
        }

        /* Take a step towards delta in transformed lambda space */
        *lambda = exp(log(*lambda + 1) + direction * tau) - 1;
        weighted_pdas(n, y, wi, *lambda, x, z, &iter, p, m, delta_s, delta_e, maxiter, verbose);
        *iters += iter;
        mse_prev = mse;
        evaluate_search_point(n, y, wi, delta, x, &mse, &err);

        /* Interpolate to next search point in transformed lambda space */
        tau_interp = tau * direction * (delta - mse) / (mse - mse_prev);
        *lambda = exp(log(*lambda + 1) + tau_interp) - 1;
        weighted_pdas(n, y, wi, *lambda, x, z, &iter, p, m, delta_s, delta_e, maxiter, verbose);
        *iters += iter;
        evaluate_search_point(n, y, wi, delta, x, &mse, &err);
        direction = sign(delta - mse);

    }
 
    /************************** Stepping Phase **************************/
           
    /* Step towards delta until we cross or land in tol band */
    while ((direction * sign(delta - mse) > 0) && (err > tol)){
        
        /* Take a step towards delta in transformed lambda space */
        *lambda = exp(log(*lambda + 1) + direction * tau) - 1;
        weighted_pdas(n, y, wi, *lambda, x, z, &iter, p, m, delta_s, delta_e, maxiter, verbose);
        *iters += iter;
        mse_prev = mse;
        evaluate_search_point(n, y, wi, delta, x, &mse, &err);
        
    }
    
    /************************** Refine Estimate **************************/

    /* Interpolate to delta in transformed lambda space to refine estimate */
    tau_interp = tau * direction * (delta - mse) / (mse - mse_prev);
    *lambda = exp(log(*lambda + 1) + tau_interp) - 1;
    weighted_pdas(n, y, wi, *lambda, x, z, &iter, p, m, delta_s, delta_e, maxiter, verbose);
    *iters += iter;
    return 1;
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
