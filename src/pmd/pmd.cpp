#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mkl.h>
#include "../proxtf/line_search.h"
#include "../utils/welch.h"
#include <proxtv.h>
#include <algorithm>

/*----------------------------------------------------------------------------*
 *--------------------------- Generic Helper Funcs ---------------------------*
 *----------------------------------------------------------------------------*/


/* Compute ||x - y||_2 where x,y are len n arrs. The last array arg y is 
 * modified inplace for the difference operation prior to taking norm.
 * 
 * Optimization Notes: 
 *      Could be made inline. 
 *      vdSub suboptimal for small (n<40) arrays. 
 */
double distance_inplace(const MKL_INT n, 
                        const double* x,
                        double* y)
{
    /* y <- x - y */
    vdSub(n, x, y, y);

    /* return ||y||_2 */
    return cblas_dnrm2(n, y, 1);
}


/* Copy leading n vals of arr source leading n vals of array dest
 *
 * Optimization Notes: 
 *      Should be made inline. 
 */
void copy(const MKL_INT n, 
          const double* source,
          double* dest)
{
    cblas_dcopy(n, source, 1, dest, 1);
}


/* Normlizes len n array x by ||x||_2 in place */
void normalize(const MKL_INT n, double* x)
{    
    /* norm <- ||x||_2 */
    double norm = cblas_dnrm2(n, x, 1);
    
    /* x /= norm */
    if (norm > 0) {
        cblas_dscal(n, 1 / norm, x, 1);
    } else {
        return ; // Array Already == 0 Everywhere
    }
}


/* Intializes len n constant vector */
void initvec(const MKL_INT n, double* x, const double val)
{
    MKL_INT i;
    for(i = 0; i < n; i++)
        x[i]= val;
}

/*----------------------------------------------------------------------------*
 *--------------------------- Spatial Helper Funcs ---------------------------*
 *----------------------------------------------------------------------------*/


/* Updates & normalizes the spatial component u_k in place by regressing 
 * the current temporal component v_k against the current residual R_k
 */
void regress_spatial(const MKL_INT d,
                     const MKL_INT t,
                     const double* R_k, 
                     double* u_k, 
                     const double* v_k)
{ 
    /* u = Yv */
    cblas_dgemv(CblasColMajor, CblasNoTrans, d, t, 1.0, R_k, d, v_k, 1, 0.0, u_k, 1); 
    
    /* u /= ||u||_2 */ 
    normalize(d, u_k);
}


/* Denoises and normalizes the  d1xd2 spatial component u_k using the 
 * proxTV douglass rachford splitting cpp implementation of TV denoising.
 */
void denoise_spatial(const MKL_INT d1,
                     const MKL_INT d2,
                     double* u_k,
                     const double lambda_tv)
{

    /* Declar & Initialize Internal Variables */
    double* target = (double *) malloc(d1 * d2 * sizeof(double));
    double* info = (double *) malloc(3 * sizeof(double));
    copy(d1*d2, u_k, target);

    /* u_k <- argmin_u ||u_k - u|| + 2*lambda_tv ||u||TV */
    DR2_TV(d1, d2, target, lambda_tv, lambda_tv, 1, 1, u_k, 1, 1, info);

    /* u_k /= ||u_k|| */
    normalize(d1*d2, u_k);

    /* Free Allocated Memory */
    free(target);
    free(info);
}


/* Update spatial component by regression of temporal component against
 * the residual followed by denoising via TV prox operator. Returns the 
 * normed difference between the updated spatial component and the 
 * previous iterate (used to monitor convergence).
 */
double update_spatial(const MKL_INT d1,
                      const MKL_INT d2,
                      const MKL_INT t,
                      const double *R_k,
                      double* u_k,
                      const double* v_k,
                      const double lambda_tv)
{
    /* Declare & Allocate For Internal Vars */
    MKL_INT d = d1*d2;
    double delta_u;
    double* u__ = (double *) malloc(d * sizeof(double));

    /* u__ <- u_k */
    copy(d, u_k, u__);

    /* u_{k+1} <- R_{k+1} v_k */
    regress_spatial(d, t, R_k, u_k, v_k);

    /* u_{k+1} <- argmin_u ||u_{k+1} - u||_2^2 + 2* lambda_tv ||u||_TV */
    denoise_spatial(d1, d2, u_k, lambda_tv);

    /* delta_u <- ||u_{k+1} - u_{k}||_2 */
    delta_u = distance_inplace(d, u_k, u__);
   
    /* Free Allocated Memory */ 
    free(u__);
    return delta_u;
}


/*----------------------------------------------------------------------------*
 *--------------------------- Temporal Helper Funcs --------------------------*
 *----------------------------------------------------------------------------*/


/* Updates the temporal component v_k in place by regressing the transpose
 * of the current temporal component u_k against the current residual R_k'
 */
void regress_temporal(const MKL_INT d,
                      const MKL_INT t,
                      const double* R_k, 
                      const double* u_k, 
                      double* v_k)
{
    /* v = R_k'u */
    cblas_dgemv(CblasColMajor, CblasTrans, d, t, 1.0, R_k, d, u_k, 1, 0.0, v_k, 1); 
    
    /* Skip Normalization */
    return ;
}


/* Denoises and normalizes the len t spatial component v_k using the 
 * constrained PDAS implementation of 2nd order L1TF denoising.
 */
void denoise_temporal(const MKL_INT t,
                      double* v_k,
                      double* z_k,
                      double* lambda_tf,
                      void* FFT=NULL)
{
    /* Declare & Allocate Local Variables */
    int iters;
    short status;
    double scale, tau, delta;
    double* wi = (double *) malloc(t * sizeof(double));
    double *v = (double *) malloc(t * sizeof(double));

    /* Initialize Local Variables */
    iters = 0;
    initvec(t, wi, 1.0);  // Constant Weight For Now
    copy(t, v_k, v); // target for tf

    /* Estimate Noise Level Via Welch PSD Estimate & Compute Ideal Step Size */
    delta = psd_noise_estimate(t, v_k, FFT);
    //scale = compute_scale(t, v_k, delta);
    //tau = (log(20 + (1 / scale)) - log(3 + (1 / scale))) / 60;

    /* If Uninitialized Compute Starting Point For Search */
    if (*lambda_tf <= 0){
        scale = compute_scale(t, v_k, delta);
        *lambda_tf = exp((log(20+(1/scale)) - log(3+(1/scale))) / 2 + log(3*scale + 1)) - 1;
    }

    /* v_k <- argmin_{v_k} ||v_k||_TF s.t. ||v - v_k||_2^2 <= T * delta */
    //status = line_search(t, v, wi, delta, tau, v_k, z_k, lambda_tf, &iters, 1, 1e-3, 0);
    status = cps_tf(t, v, wi, delta, v_k, z_k, lambda_tf, &iters, 2e-2, 0);

    if (status < 1){
        copy(t, v, v_k);
    }

    /* v_k /= ||v_k||_2 */
    normalize(t, v_k);

    /* Free Allocated Memory */
    free(v);
    free(wi);
}


/* Update temporal component by regression of spatial component against
 * the residual followed by denoising via constrained TF. Returns the 
 * normed difference between the updated temporal component and the 
 * previous iterate (used to monitor convergence).
 */
double update_temporal(const MKL_INT d,
                       const MKL_INT t,
                       const double* R_k, 
                       const double* u_k,
                       double* v_k,
                       double* z_k,
                       double* lambda_tf,
                       void* FFT=NULL)
{
    /* Declare & Allocate For Internal Vars */
    double delta_v;
    double* v__ = (double *) malloc(t * sizeof(double));

    /* v__ <- v_k */
    copy(t, v_k, v__);

    /* v_{k+1} <- R_{k+1}' u_k */
    regress_temporal(d, t, R_k, u_k, v_k);

    /* v_{k+1} <- argmin_v ||v||_TF s.t. ||v_{k+1} - v||_2^2 <= T * delta */
    denoise_temporal(t, v_k, z_k, lambda_tf, FFT);

    /* return ||v_{k+1} - v_{k}||_2 */
    delta_v = distance_inplace(t, v_k, v__);
   
    /* Free Allocated Memory */
    free(v__); 
    return delta_v;
}

/*----------------------------------------------------------------------------*
 *-------------------------- Inner Loop (Seq) Funcs --------------------------*
 *----------------------------------------------------------------------------*/


/* Computes the ratio of the L1 to TV norm for a spatial components in order to 
 * test against the null hypothesis that it is gaussian noise.
 */
double spatial_test_statistic(const MKL_INT d1,
                              const MKL_INT d2,
                              const double* u_k)
{
    
    /* Declare & Initialize Internal Variables */
    int j, j1, j2;
    double norm_tv = 0;
    double norm_l1 = u_k[d1*d2-1];  // Bottom, Right Corner

    /* All Elements Except Union (Bottom Row, Rightmost Column) */
    for (j2 = 0; j2 < d2-1; j2++){
        for (j1 = 0; j1 < d1-1; j1++){
            j = d1 * j2 + j1;
            norm_tv += fabs(u_k[j] - u_k[j+1]) + fabs(u_k[j] - u_k[j + d1]);
            norm_l1 += fabs(u_k[j]);
        }
    }

    /* Rightmost Column (Skip Bottom Element) */
    j = d1 * (d2-1);
    for (j1 = 0; j1 < d1 - 1; j1++){
        norm_tv += fabs(u_k[j] - u_k[j+1]);
        norm_l1 += fabs(u_k[j]);
        j += 1;
    }

    /* Bottom Row (Skip Rightmost Element) */
    j = d1 - 1;
    for (j2 = 0; j2 < d2-1; j2++){
        norm_tv += fabs(u_k[j] - u_k[j + d1]);
        norm_l1 += fabs(u_k[j]);
        j += d1;
    }
    
    //return norm_tv / (d1 * (d2 - 1) + d2 * (d1- 1));
    /* Return Test Statistic */
    if (norm_tv > 0){
        return norm_l1 / norm_tv;
    } else{
        return 0; // Spatial component constant => garbage
    }
}

/* Computes the ratio of the L1 to TV norm for a spatial components in order to 
 * test against the null hypothesis that it is gaussian noise.
 */
double temporal_test_statistic(const MKL_INT t,
                               const double* v_k)
{
    
    /* Declare & Initialize Internal Variables */
    int i;
    double norm_tf = 0;
    double norm_l1 = fabs(v_k[0]) + fabs(v_k[t-1]);

    /* All Elements Except Union (Bottom Row, Rightmost Column) */
    for (i = 1; i < t-1; i++){
        norm_tf += fabs(v_k[i]+ v_k[i] - v_k[i-1] - v_k[i+1]); 
        norm_l1 += fabs(v_k[i]);
    }

    /* Return Test Statistic */
    if (norm_l1 > 0){
        return norm_tf / norm_l1;
    } else{
        return 0; // temporal component constant => garbage
    }
}

/* Intialize with equivalent of temporal upate where u_k = 1/sqrt(d).
 * Return computed value of lambda_tf to use as a warm start.
 */
double initialize_components(const MKL_INT d,
                             const MKL_INT t,
                             const double* R_k,
                             double* u_k,
                             double* v_k,
                             double* z_k,
                             void* FFT=NULL){
    
    /* Declare Internal Variables */
    double lambda_tf = 0;

    /* Initialize Spatial To Constant Vector & Warm Start To 0 */
    initvec(d, u_k, 1 / sqrt(d));
    initvec(t-2, z_k, 0.0);

    /* v_k <- R_k' u_k */
    regress_temporal(d, t, R_k, u_k, v_k);
 
    /* v_k <- argmin_v ||v||_TF s.t. ||v_k - v||_2^2 <= delta * T */
    denoise_temporal(t, v_k, z_k, &lambda_tf, FFT);

    return lambda_tf;
}


/* Solves: 
 *  u_k, v_k : min <u_k, R_k v_k> - lambda_tv ||u_k||_TV - lambda_tf ||v_k||_TF
 *  
 *  Returns:
 *      1: If we reject the null hypothesis that u_k is noise 
 *     -1: If we accept the null hypothesis that u_k is noise
 */
int rank_one_decomposition(const MKL_INT d1, 
                           const MKL_INT d2, 
                           const MKL_INT t,
                           const double* R_k, 
                           double* u_k, 
                           double* v_k,
                           const double lambda_tv,
                           const double spatial_thresh,
                           const MKL_INT max_iters,
                           const double tol,
                           void* FFT=NULL)
{
 
    /* Declare & Allocate Mem For Internal Vars */
    MKL_INT d, iters;
    double delta_u, delta_v, lambda_tf;
    double *z_k = (double *) malloc((t-2) * sizeof(double));

    /* Initialize Internal Variables */
    d = d1 * d2;
    lambda_tf = initialize_components(d, t, R_k, u_k, v_k, z_k, FFT);

    /* Loop Until Convergence Of Spatial & Temporal Components */
    for (iters = 0; iters < max_iters; iters++){
        
        /* Update Components */
        delta_u = update_spatial(d1, d2, t, R_k, u_k, v_k, lambda_tv);
        delta_v = update_temporal(d, t, R_k, u_k, v_k, z_k, &lambda_tf, FFT);

        /* Check Convergence */
        if (fmax(delta_u, delta_v) < tol){    
            /* Free Allocated Memory & Test Spatial Component Against Null */
            free(z_k);
            if (spatial_test_statistic(d1, d2, u_k) < spatial_thresh) 
                return -1;  // Discard Component
            return 1;  // Keep Component
        }

        /* Preemptive Check To See If We're Fitting Noise */
        if (iters == 9){
            if (spatial_test_statistic(d1, d2, u_k) < spatial_thresh){         
                /* Free Allocated Memory & Return */
                free(z_k); 
                return -1; // Discard Component
            } 
        }    
    }

    /* MAXITER EXCEEDED: Free Memory & Test Spatial Component Against Null */
    free(z_k);
    if (spatial_test_statistic(d1, d2, u_k) < spatial_thresh){ 
        return -1;  // Discard Component
    }
    return 1;  // Keep Component
}


/* Apply TF/TV Penalized Matrix Decomposition (PMD) to factor a (d1*d2)xT
 * column major formatted video into sptial and temporal components.
 */
size_t pmd(const MKL_INT d1, 
           const MKL_INT d2, 
           const MKL_INT t,
           double* R, 
           double* U,
           double* V,
           const double lambda_tv,
           const double spatial_thresh,
           const size_t max_components,
           const size_t consec_failures,
           const size_t max_iters,
           const double tol,
           void* FFT=NULL)  /* Handle Provided For Threadsafe FFT */
{
    /* Declare & Intialize Internal Vars */
    int* keep_flag = (int *) malloc(consec_failures*sizeof(int));
    int max_keep_flag;
    size_t i, k, good = 0;
    MKL_INT d = d1 * d2;

    /* Fill Keep Flags */
    for (i = 0; i < consec_failures; i++) keep_flag[i] = 1;

    /* Sequentially Extract Rank 1 Updates Until We Are Fitting Noise */
    for (k = 0; k < max_components; k++, good++) {
        
        /* U[:,k] <- u_k, V[k,:] <- v_k' : 
         *    min <u_k, R_k v_k> - lambda_tv ||u_k||_TV - lambda_tf ||v_k||_TF
         */
        keep_flag[k % consec_failures] = rank_one_decomposition(d1, d2, t, R, 
                                                                U + good*d, 
                                                                V + good*t, 
                                                                lambda_tv, 
                                                                spatial_thresh, 
                                                                max_iters, 
                                                                tol, FFT);

        /* Check Component Quality: Terminate if we're fitting noise */
        if (keep_flag[k % consec_failures] < 0){
            max_keep_flag = -1;  // current component is a failure
            for (i=1; i < consec_failures; i++){  // check previous
                max_keep_flag = max(max_keep_flag, keep_flag[(k+i) % consec_failures]);
            } 
            if (max_keep_flag < 0){
                free(keep_flag);
                return good;
            }
        }
        /* Debias Components */
        regress_temporal(d, t, R, U + good*d, V + good*t);

        /* Update Residual: R_k <- R_k - U[:,k] V[k,:] */
        cblas_dger(CblasColMajor, d, t, -1.0, U + good*d, 1, V + good*t, 1, R, d);

        /* Make Sure We Overwrite Failed Components */
        if (keep_flag[k % consec_failures] < 0) good--;
    }

    /* MAXCOMPONENTS EXCEEDED: Terminate Early */
    free(keep_flag);
    return good; 
    return k;
}


/* Wrap TV/TF Penalized Matrix Decomposition with OMP directives to enable parallel, 
 * block-wiseprocessing of large datasets in shared memory.
 */
void batch_pmd(const MKL_INT bheight, 
               const MKL_INT bwidth, 
               const MKL_INT t,
               const MKL_INT b,
               double** Rpt, 
               double** Upt,
               double** Vpt,
               size_t* Kpt,
               const double lambda_tv,
               const double spatial_thresh,
               const size_t max_components,
               const size_t consec_failures,
               const size_t max_iters,
               const double tol)
{
    // Create FFT Handle So It can Be Shared Aross Threads
    DFTI_DESCRIPTOR_HANDLE FFT;
    MKL_LONG status;
    MKL_LONG L = 256;  /* Size Of Subsamples in welch estimator */
    status = DftiCreateDescriptor( &FFT, DFTI_DOUBLE, DFTI_REAL, 1, L );
    status = DftiSetValue( FFT, DFTI_PACKED_FORMAT, DFTI_PACK_FORMAT);
    status = DftiCommitDescriptor( FFT );
    if (status != 0) 
        fprintf(stderr, "Error while creating MKL_FFT Handle: %ld\n", status);

    // Loop Over All Patches In Parallel
    int m;
    #pragma omp parallel for shared(FFT) schedule(guided)
    for (m = 0; m < b; m++){
        //Use dummy vars for decomposition  
        Kpt[m] = pmd(bheight, bwidth, t, Rpt[m], Upt[m], Vpt[m], lambda_tv, 
                     spatial_thresh, max_components, consec_failures, max_iters, 
                     tol, &FFT);
    }

    // Free MKL FFT Handle
    status = DftiFreeDescriptor( &FFT ); 
    if (status != 0)
        fprintf(stderr, "Error while deallocating MKL_FFT Handle: %ld\n", status);
}
