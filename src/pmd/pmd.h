#ifndef PMD_H
#define PMD_H

#include <mkl.h>
#include <mkl_dfti.h>

double distance_inplace(const MKL_INT n, 
                        const double* x,
                        double* y);

void copy(const MKL_INT n, 
          const double* source,
          double* dest);

void normalize(const MKL_INT n, double* x);

void initvec(const MKL_INT n, double* x, const double val);

void regress_spatial(const MKL_INT d,
                     const MKL_INT t,
                     const double* R_k, 
                     double* u_k, 
                     const double* v_k);

void constrained_denoise_spatial(const MKL_INT d1,
                                 const MKL_INT d2,
                                 double* u_k,
                                 double* lambda_tv);

void denoise_spatial(const MKL_INT d1,
                     const MKL_INT d2,
                     double* u_k,
                     double *lambda_tv);

void constrained_denoise_spatial(const MKL_INT d1,
                                 const MKL_INT d2,
                                 double* u_k,
                                 double* lambda_tv);

double update_spatial_init(const MKL_INT d,
                           const MKL_INT t,
                           const double *R_k,
                           double* u_k,
                           const double* v_k);

double update_spatial(const MKL_INT d1,
                      const MKL_INT d2,
                      const MKL_INT t,
                      const double *R_k,
                      double* u_k,
                      const double* v_k,
                      double* lambda_tv);

void regress_temporal(const MKL_INT d,
                      const MKL_INT t,
                      const double* R_k, 
                      const double* u_k, 
                      double* v_k);

double compute_scale(const MKL_INT t, 
                     const double *y, 
                     const double delta);

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
                       const MKL_INT t,
                       const double* R_k, 
                       const double* u_k,
                       double* v_k,
                       double* z_k,
                       double* lambda_tf,
                       void* FFT=NULL);

double spatial_test_statistic(const MKL_INT d1,
                              const MKL_INT d2,
                              const double* u_k);

double temporal_test_statistic(const MKL_INT t,
                               const double* v_k);

void init_dual_from_primal(const MKL_INT t, 
                           const double* v,
                           double* z);

int rank_one_decomposition(const MKL_INT d1, 
                           const MKL_INT d2, 
                           const MKL_INT d_sub,
                           const MKL_INT t,
                           const MKL_INT t_sub,
                           const double* R_k, 
                           const double* R_init, 
                           double* u_k, 
                           double* u_init, 
                           double* v_k,
                           double* v_init,
                           const double spatial_thresh,
                           const double temporal_thresh,
                           const MKL_INT max_iters_main,
                           const MKL_INT max_iters_init,
                           const double tol,
                           void* FFT=NULL);

size_t pmd(const MKL_INT d1, 
           const MKL_INT d2, 
           MKL_INT d_sub, 
           const MKL_INT t,
           MKL_INT t_sub,
           double* R, 
           double* R_ds,
           double* U,
           double* V,
           const double spatial_thresh,
           const double temporal_thresh,
           const size_t max_components,
           const size_t consec_failures,
           const MKL_INT max_iters_main,
           const MKL_INT max_iters_init,
           const double tol,
           void* FFT=NULL);

void batch_pmd(const MKL_INT bheight,
               const MKL_INT bwidth, 
               MKL_INT d_sub,
               const MKL_INT t,
               MKL_INT t_sub,
               const int b,
               double** Rpt, 
               double** Rpt_ds, 
               double** Upt,
               double** Vpt,
               size_t* K,
               const double spatial_thresh,
               const double temporal_thresh,
               const size_t max_components,
               const size_t consec_failures,
               const size_t max_iters,
               const size_t max_iters_ds,
               const double tol);
#endif /* PMD_H */
