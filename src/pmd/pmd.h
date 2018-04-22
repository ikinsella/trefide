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

void denoise_spatial(const MKL_INT d1,
                     const MKL_INT d2,
                     double* u_k,
                     const double lambda_tv);

double update_spatial(const MKL_INT d1,
                      const MKL_INT d2,
                      const MKL_INT t,
                      const double *R_k,
                      double* u_k,
                      const double* v_k,
                      const double lambda_tv);

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

double initialize_components(const MKL_INT d,
                             const MKL_INT t,
                             const double* R_k,
                             double* u_k,
                             double* v_k,
                             double* z_k,
                             void* FFT=NULL);

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
                           void* FFT=NULL);

size_t pmd(const MKL_INT d1, 
           const MKL_INT d2, 
           const MKL_INT t,
           double* R, 
           double* U,
           double* V,
           const double lambda_tv,
           const double spatial_thresh,
           const size_t max_components,
           const size_t max_iters,
           const double tol,
           void* FFT=NULL);

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
               const size_t max_iters,
               const double tol);

#endif /* PMD_H */
