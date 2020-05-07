#ifndef PMD_H
#define PMD_H

#include <mkl.h>
#include <mkl_dfti.h>
#include "../proxtf/utils.h"

#define FORCEINLINE __attribute__((always_inline)) inline

class PMD_params {
private:

    const MKL_INT bheight;
    const MKL_INT bwidth;
    MKL_INT d_sub;
    const MKL_INT t;
    MKL_INT t_sub;
    const double spatial_thresh;
    const double temporal_thresh;
    const size_t max_components;
    const size_t consec_failures;
    const size_t max_iters_main;
    const size_t max_iters_init;
    const double tol;
    void* FFT; /* Handle Provided For Threadsafe FFT */
    bool enable_temporal_denoiser;
    bool enable_spatial_denoiser;

public:

    PMD_params(
            const MKL_INT _bheight,
            const MKL_INT _bwidth,
            MKL_INT _d_sub,
            const MKL_INT _t,
            MKL_INT _t_sub,
            const double _spatial_thresh,
            const double _temporal_thresh,
            const size_t _max_components,
            const size_t _consec_failures,
            const size_t _max_iters_main,
            const size_t _max_iters_init,
            const double _tol,
            void* _FFT,
            bool _enable_temporal_denoiser,
            bool _enable_spatial_denoiser);

    MKL_INT get_bheight();
    MKL_INT get_bwidth();
    MKL_INT get_d_sub();
    void set_d_sub(MKL_INT _d_sub);
    MKL_INT get_t();
    MKL_INT get_t_sub();
    void set_t_sub(MKL_INT _t_sub);
    double get_spatial_thresh();
    double get_temporal_thresh();
    size_t get_max_components();
    size_t get_consec_failures();
    size_t get_max_iters_main();
    size_t get_max_iters_init();
    double get_tol();
    void* get_FFT();
    void set_FFT(void* _FFT);
    bool get_enable_temporal_denoiser();
    void set_enable_temporal_denoiser(bool _enable_temporal_denoiser);
    bool get_enable_spatial_denoiser();
    void set_enable_spatial_denoiser(bool _enable_spatial_denoiser);
};


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
inline double distance_inplace(const MKL_INT n, const double* x, double* y)
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
inline void copy(const MKL_INT n, const double* source, double* dest)
{
    cblas_dcopy(n, source, 1, dest, 1);
}


/* Normlizes len n array x by ||x||_2 in place */
inline void normalize(const MKL_INT n, double* x)
{
    /* norm <- ||x||_2 */
    double norm = cblas_dnrm2(n, x, 1);

    /* x /= norm */
    if (norm > 0) {
        cblas_dscal(n, 1 / norm, x, 1);
    }
    // Otherwise return, since Array Already == 0 Everywhere
}


/* Intializes len n constant vector */
inline void initvec(MKL_INT n, double* x, const double val)
{
    while (n--)
        *x++ = val;
}

/* End generic helper functions */


/* Updates & normalizes the spatial component u_k in place by regressing
 * the current temporal component v_k against the current residual R_k
 */
inline void regress_spatial(const MKL_INT d, const MKL_INT t,
                                 const double* R_k, double* u_k,
                                 const double* v_k)
{
    /* u = Yv */
    cblas_dgemv(CblasColMajor, CblasNoTrans, d, t, 1.0, R_k, d, v_k, 1, 0.0, u_k, 1);

    /* u /= ||u||_2 */
    normalize(d, u_k);  // Temporarily Removed For Constrained TV Testing
}

void constrained_denoise_spatial(const MKL_INT d1,
                                 const MKL_INT d2,
                                 double* u_k,
                                 double* lambda_tv);

void denoise_spatial(const MKL_INT d1,
                     const MKL_INT d2,
                     double* u_k,
                     double *lambda_tv);

double update_spatial_init(const MKL_INT d,
                           const MKL_INT t,
                           const double *R_k,
                           double* u_k,
                           const double* v_k);

double update_spatial(const double *R_k,
                      double* u_k,
                      const double* v_k,
                      double *lambda_tv,
                      PMD_params *pars);

/* Updates the temporal component v_k in place by regressing the transpose
 * of the current temporal component u_k against the current residual R_k'
 */
inline void regress_temporal(const MKL_INT d, const MKL_INT t,
                                  const double* R_k, const double* u_k,
                                  double* v_k)
{
    /* v = R_k'u */
    cblas_dgemv(CblasColMajor, CblasTrans, d, t, 1.0, R_k, d, u_k, 1, 0.0, v_k, 1);

    /* Skip Normalization */
}

void regress_temporal(const MKL_INT d,
                      const MKL_INT t,
                      const double* R_k,
                      const double* u_k,
                      double* v_k);

void denoise_temporal(const MKL_INT t,
                      double* v_k,
                      double* z_k,
                      double* lambda_tf,
                      void* FFT=NULL);

double update_temporal_init(const MKL_INT d,
                            const MKL_INT t,
                            const double* R_k,
                            const double* u_k,
                            double* v_k);

double update_temporal(const MKL_INT d,
                       const double* R_k,
                       const double* u_k,
                       double* v_k,
                       double* z_k,
                       double* lambda_tf,
                       PMD_params *pars);

double spatial_test_statistic(const MKL_INT d1,
                              const MKL_INT d2,
                              const double* u_k);

double temporal_test_statistic(const MKL_INT t, const double* v_k);

/* Initialize A Partition Of The Dual TF Var From A Primal TF Var
 */
inline void init_dual_from_primal(const MKL_INT t, const double* v,
                                       double* z)
{
    /* Compute Second Order Differences */
    Dx(t, v, z);

    /* Partition Elements Based On Differences */
    for (int j = 0; j < t-2; j++){
        if (z[j] > 1e-5){
            z[j] = 1;
        } else if (z[j] < -1e-5){
            z[j] = -1;
        } else {
            z[j] = 0;
        }
    }
}

int rank_one_decomposition(const double* R_k,
                           const double* R_init,
                           double* u_k,
                           double* u_init,
                           double* v_k,
                           double* v_init,
                           PMD_params *pars);

size_t pmd(double* R,
           double* R_ds,
           double* U,
           double* V,
           PMD_params *pars);

void batch_pmd(
        double** Rp,
        double** Rp_ds,
        double** Up,
        double** Vp,
        size_t* K,
        const int b,
        PMD_params *pars);

#endif /* PMD_H */
