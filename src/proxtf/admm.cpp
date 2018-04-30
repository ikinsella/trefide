#include <stdlib.h>
#include <math.h>
#include <mkl.h>
#include <glmgen.h>
#include "line_search.h"

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
                          const int verbose=0)
{
    /* Trend Filtering Constants */
    int DEGREE = 1;

    /* Declare & Initialize Local Variables */
    short status;
    int i;
    double scale, rho = 1;
    double * temp_n = (double *) malloc(n * sizeof(double));
    for(i = 0; i < n; i++) temp_n[i] = 1/sqrt(w[i]); // Assume w does not have zeros 
    
    /* Declare & Init Sparse Matrix Objects */ 
    cs * D = tf_calc_dk(n, DEGREE+1, x);
    cs * Dt = cs_transpose(D, 1);
    diag_times_sparse(Dt, temp_n); /* Dt = W^{-1/2} Dt */
    cs * Dk = tf_calc_dktil(n, DEGREE, x);
    cs * Dkt = cs_transpose(Dk, 1);
    cs * DktDk = cs_multiply(Dkt, Dk);
    gqr * Dt_qr = glmgen_qr(Dt);
    gqr * Dkt_qr = glmgen_qr(Dkt);

    /* If Uninitialized, Compute Starting Points For Search */
    if (*lambda <= 0){

        /* Compute Step Size wrt Transformed Lambda Space*/
        scale = compute_scale(n, y, delta);
        *lambda = exp((log(20+(1/scale)) - log(3+(1/scale))) / 2 + log(3*scale + 1)) - 1;
        /* Initialize search from maximum lambda */
        //*lambda = tf_maxlam(n, y, Dt_qr, w);
        
        /* Initialize Primal At beta_max */
        //calc_beta_max(y, w, n, Dt_qr, Dt, temp_n, beta);

        /* Check if beta = weighted mean(y) is better than beta_max */ 
        //double yc = weighted_mean(y,w,n);
        //for (i = 0; i < n; i++) temp_n[i] = yc;
        //double obj1 = tf_obj(x,y,w,n,DEGREE,*lambda,FAMILY_GAUSSIAN,beta,alpha);
        //double obj2 = tf_obj(x,y,w,n,DEGREE,*lambda,FAMILY_GAUSSIAN,temp_n,alpha);
        //if(obj2 < obj1) memcpy(beta, temp_n, n*sizeof(double));

        /* initalize alpha at alpha_max */
        //tf_dxtil(x, n, DEGREE, beta, alpha);

        /* intialize u at u_max */ 
        //for (i = 0; i < n; i++) u[i] = w[i] * (beta[i] - y[i]) / (rho * lambda[0]);
        //glmgen_qrsol (Dkt_qr, u);
    }
   
    /* Compute Rho From Data Locations */ 
    rho = rho * pow((x[n-1] - x[0])/n, (double)DEGREE);

    /* if lambda is too small, return a trivial solution */  
    if (*lambda <= 1e-10 * l1norm(y,n)/n) {
        for (i=0; i<n; i++) beta[i] = y[i];
        *lambda = 0;

        cs_spfree(D);
        cs_spfree(Dt);
        cs_spfree(Dk);
        cs_spfree(Dkt);
        cs_spfree(DktDk);
        glmgen_gqr_free(Dt_qr);
        glmgen_gqr_free(Dkt_qr);
        free(temp_n);
        return 1;
    }

    /* v_k <- argmin_{v_k} ||v_k||_TF s.t. ||v - v_k||_2^2 <= T * delta */
    status = cps_tf_admm(n, DEGREE, x, y, w, DktDk, delta, beta, alpha, u,
                         lambda, rho, iters,  tol,  verbose);

    /* Free Allocated Memory */
    cs_spfree(D);
    cs_spfree(Dt);
    cs_spfree(Dk);
    cs_spfree(Dkt);
    cs_spfree(DktDk);
    glmgen_gqr_free(Dt_qr);
    glmgen_gqr_free(Dkt_qr);
    free(temp_n);

    return status;
}


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
                  const int verbose)
{
    /* TF constants */
    int maxiter = 100;
    double obj_tol = 1e-4;

    /* Declare & allocate internal admm vars */
    int df;
    double * obj = (double *) malloc(maxiter * sizeof(double));
 
    /* Declar & allocate internal line search vars */
    double target = sqrt(n*delta);  // target norm of error
    int iter;
    double l2_err, l2_err_prev = 0;
    double * resid = (double *) malloc(n * sizeof(double));

    /* Iterate lagrangian solves over lambda; the beta, alpha, and u vectors 
     * get used for warm starts in each subsequent iteration.
     */
    while (*iters < maxiter * 100)
    {
        /* fit admm */
        tf_admm_gauss(x, y, w, n, degree, maxiter, *lambda, &df, beta, alpha, u,
                      obj, &iter, rho * (*lambda), obj_tol, DktDk, verbose);

        /* If there any NaNs in beta: reset beta, alpha, u */
        if (has_nan(beta, n)){
            free(obj);
            free(resid);
            return -1;  // abort search
        }

        /* Increment LS Total Iters */
        *iters += iter;

        /* Compute norm of residual */
        vdSub(n, y, beta, resid);
        l2_err = cblas_dnrm2(n, resid, 1);

        /* Check For Convergence */
        if (fabs(target*target - l2_err*l2_err) / (target*target) < tol){
            free(obj);
            free(resid);
            return 1; // successfully converged within tolerance
        } else if(fabs(l2_err - l2_err_prev) < 1e-3){
            free(obj);
            free(resid);
            return 0;  // Stalled before tol reached
        } 
        l2_err_prev = l2_err;

        /* Increment Lambda */
        *lambda = exp(log(*lambda) + log(target) - log(l2_err));
    }

    /* Line Search Didn't Converge */
    free(obj);
    free(resid);
    return 0; // Reached Maxiter before tol reached
}


short langrangian_tf_admm(const int n,           // data length
                          double* x,             // data locations
                          double *y,             // data observations
                          double *w,             // data observation weights
                          double lambda,        // regularization parameter
                          double *beta,          // primal variable
                          double *alpha,
                          double *u,
                          int *iter,             // pointer to iter # (so we can return it)
                          const int verbose=0)
{
    /* Trend Filtering Constants */
    int DEGREE = 1;
    int maxiter = 100;
    double obj_tol = 1e-4;

    /* Declare & allocate internal admm vars */
    int i, df;
    short status = 1;
    double rho = 1;
    double * obj = (double *) malloc(maxiter * sizeof(double));
    double * temp_n = (double *) malloc(n * sizeof(double));
    for(i = 0; i < n; i++) temp_n[i] = 1/sqrt(w[i]); // Assume w does not have zeros 
    
    /* Declare & Init Sparse Matrix Objects */ 
    cs * D = tf_calc_dk(n, DEGREE+1, x);
    cs * Dt = cs_transpose(D, 1);
    diag_times_sparse(Dt, temp_n); /* Dt = W^{-1/2} Dt */
    cs * Dk = tf_calc_dktil(n, DEGREE, x);
    cs * Dkt = cs_transpose(Dk, 1);
    cs * DktDk = cs_multiply(Dkt, Dk);
    gqr * Dt_qr = glmgen_qr(Dt);
    gqr * Dkt_qr = glmgen_qr(Dkt);

    /* Compute Rho From Data Locations */ 
    rho = rho * pow((x[n-1] - x[0])/n, (double)DEGREE);

    /* if lambda is too small, return a trivial solution */  
    if (lambda <= 1e-10 * l1norm(y,n)/n) {
        for (i=0; i<n; i++) beta[i] = y[i];
        lambda = 0;

        cs_spfree(D);
        cs_spfree(Dt);
        cs_spfree(Dk);
        cs_spfree(Dkt);
        cs_spfree(DktDk);
        glmgen_gqr_free(Dt_qr);
        glmgen_gqr_free(Dkt_qr);
        free(obj);
        free(temp_n);
        return status;
    }

    /* fit admm */
    tf_admm_gauss(x, y, w, n, DEGREE, maxiter, lambda, &df, beta, alpha, u,
                  obj, iter, rho * lambda, obj_tol, DktDk, verbose);

    /* If there any NaNs in beta: solve failed */
    if (has_nan(beta, n)){
        status = -1;
    }

    /* Free Allocated Memory */
    cs_spfree(D);
    cs_spfree(Dt);
    cs_spfree(Dk);
    cs_spfree(Dkt);
    cs_spfree(DktDk);
    glmgen_gqr_free(Dt_qr);
    glmgen_gqr_free(Dkt_qr);
    free(obj);
    free(temp_n);

    return status;
}
