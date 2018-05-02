#ifndef DECIMATION_H
#define DECIMATION_H
#include <mkl.h>
void downsample_1d(const int t, 
                   const int ds, 
                   const double* v, 
                   double* v_ds);


void downsample_2d(const int d1, 
                   const int d2, 
                   const int ds, 
                   const double* u, 
                   double* u_ds);

void downsample_3d(const int d1, 
                   const int d2, 
                   const int d_sub, 
                   const int t, 
                   const int t_sub, 
                   double *Y, 
                   double *Y_ds);

void upsample_1d_inplace(const int t, 
                         const int ds, 
                         double* v);


void upsample_1d(const int t, 
                 const int ds, 
                 double* v, 
                 const double* v_ds);

void upsample_2d(const int d1, 
                 const int d2, 
                 const int ds, 
                 double* u, 
                 double* u_ds);

int decimated_rod(const MKL_INT d1, 
                  const MKL_INT d1_ds, 
                  const MKL_INT d2, 
                  const MKL_INT d2_ds, 
                  const MKL_INT t,
                  const MKL_INT t_ds,
                  const double* R_k, 
                  const double* R_ds, 
                  double* u_k, 
                  double* u_ds, 
                  double* v_k,
                  double* v_ds,
                  const double lambda_tv,
                  const double spatial_thresh,
                  const MKL_INT max_iters,
                  const MKL_INT max_iters_ds,
                  const double tol,
                  void* FFT=NULL);

size_t decimated_pmd(const MKL_INT d1, 
                     const MKL_INT d1_ds,
                     const MKL_INT d2, 
                     const MKL_INT d2_ds,
                     const MKL_INT t,
                     const MKL_INT t_ds,
                     double* R, 
                     double* R_ds,
                     double* U,
                     double* V,
                     const double lambda_tv,
                     const size_t max_components,
                     const size_t consec_failures,
                     const size_t max_iters,
                     const size_t max_iters_ds,
                     const double tol,
                     void* FFT=NULL);

void decimated_batch_pmd(const MKL_INT bheight,
                         const MKL_INT bheight_ds, 
                         const MKL_INT bwidth, 
                         const MKL_INT bwidth_ds, 
                         const MKL_INT t,
                         const MKL_INT t_ds,
                         const MKL_INT b,
                         double** R, 
                         double** R_ds, 
                         double** U,
                         double** V,
                         size_t* K,
                         const double lambda_tv,
                         const size_t max_components,
                         const size_t consec_failures,
                         const size_t max_iters,
                         const size_t max_iters_ds,
                         const double tol);
#endif /* DECIMATION_H */
