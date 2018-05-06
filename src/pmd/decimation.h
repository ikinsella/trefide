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
                   const double *Y, 
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
#endif /* DECIMATION_H */
