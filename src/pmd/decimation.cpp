#include <stdlib.h>
#include <stdio.h>
#include <mkl.h>
#include <math.h>
#include <algorithm>

/*----------------------------------------------------------------------------*
 *-------------------------- Downsampling Routines ---------------------------*
 *----------------------------------------------------------------------------*/


/* Compute downsampled signal as average in each (ds, ) neighborhood */
void downsample_1d(const int t, 
                   const int ds, 
                   const double* v, 
                   double* v_ds)
{
    int t_ds = t / ds;
    int i_ds;
    for (i_ds = 0; i_ds < t_ds; i_ds++){
        int i;
        v_ds[i_ds] = 0;
        for (i = 0; i < ds; i++){
            v_ds[i_ds] += v[i_ds * ds + i];
        }
        v_ds[i_ds] /= ds;
    }
}


/* Compute downsampled image pixels as average in each (ds,ds) neighborhood*/
void downsample_2d(const int d1, 
                   const int d2, 
                   const int ds, 
                   const double* u, 
                   double* u_ds)
{
    int d2_ds = d2 / ds;
    int d1_ds = d1 / ds;
    int j_ds;
    for (j_ds = 0; j_ds < d2_ds; j_ds++){
        int i_ds;
        for (i_ds = 0; i_ds < d1_ds; i_ds++){
            u_ds[i_ds + d1_ds * j_ds] = 0;
            int j;
            for (j = 0; j < ds; j++){
                int i;
                for (i = 0; i < ds; i++){
                    u_ds[i_ds + d1_ds * j_ds] += u[(i_ds*ds + i) + d1 * (j_ds * ds + j)];
                }
            }
            u_ds[i_ds + d1_ds * j_ds] /= ds * ds;
        }
    }
}


/* Downsample (d1xd2xt) movie in row-major order
 */
void downsample_3d(const int d1, 
                   const int d2, 
                   const int d_sub, 
                   const int t, 
                   const int t_sub, 
                   const double *Y, 
                   double *Y_ds)
{

    /* Declare & Initialize Local Variables */
    int d = d1 * d2;
    int d1_ds = d1 / d_sub;
    int d2_ds = d2 / d_sub;
    int d_ds = d1_ds * d2_ds;
    int t_ds = t / t_sub;
    double elems_per_block = d_sub * d_sub * t_sub;

    /* Init Downsampled Elems to 0*/
    int k;
    for (k = 0; k < t_ds; k++){
        int j;
        for (j = 0; j < d2_ds; j++){
            int i;
            for (i = 0; i < d1_ds; i++){
                Y_ds[i + d1_ds*j + d_ds*k] = 0;
            }
        }
    }

    /* Loop Over Elems of Y in order of storage */
    for (k = 0; k < t; k++){
        int j;
        for (j = 0; j < d2; j++){
            int i;
            for (i = 0; i < d1; i++){
                Y_ds[(i/d_sub)+(j/d_sub)*d1_ds+(k/t_sub)*d_ds] += Y[i + d1*j + d*k] / elems_per_block;
            }
        }
    }
}  


/*----------------------------------------------------------------------------*
 *-------------------------- Upsampling Routines -----------------------------*
 *----------------------------------------------------------------------------*/


/* Linearly Interpolate Between Missing Values In 1D Upsampled Array
 */
void upsample_1d_inplace(const int t, 
                         const int ds, 
                         double* v)
{
    /* Declare & Initialize Local Variables */
    int i, i_ds, t_ds = t / ds;
    double start, end = v[0];

    /* Fill Missing Elements Between Observed Samples*/
    for (i_ds = 0; i_ds < t_ds - 2; i_ds++){
        start = end;
        end = v[(i_ds + 1) * ds];
        for (i = 1; i < ds; i++) v[i_ds * ds + i] = start  + (i / ds) * (end - start); 
    }

    /* Continue Linear Trend For Trailing Elements */
    end = v[t - ds] + end - start; 
    start = v[t - ds];  
    for (i = 1; i < ds; i++) v[t - ds + i] = start  + (i / ds) * (end - start);
}


/* Fill Upsampled 1D Array From Downsampled 1D Array Via Linear Interpolation
 */
void upsample_1d(const int t, 
                 const int ds, 
                 double* v, 
                 const double* v_ds)
{
    /* Declare & Initialize Local Variables */
    int t_ds = t / ds;

    /* Copy Elements Of DS Array Into Respective Locs In US Array */
    cblas_dcopy(t_ds, v_ds, 1, v, ds);
    
    /* Upsample Missing Elements Of US Array */
    upsample_1d_inplace(t, ds, v);
}


/* Fill Upsampled 2D Array From Downsampled 2D Array Via Bilinear Interpolation
 */
void upsample_2d(const int d1, 
                 const int d2, 
                 const int ds, 
                 double* u, 
                 double* u_ds)
{
    /* Declare & Initialize Local Variables */
    int d2_ds = d2 / ds;
    int d1_ds = d1 / ds;

    /* Perform Inplace Transpose To Get Row Major Formatted Image */
    mkl_dimatcopy('C', 'T', d1, d2, 1.0, u, d1, d2);
    mkl_dimatcopy('C', 'T', d1_ds, d2_ds, 1.0, u_ds, d1_ds, d2_ds);

    /* Start By Interpolating Along Rows Where We Have Samples */
    int i_ds;
    for (i_ds = 0; i_ds < d1_ds; i_ds++) 
        upsample_1d(d2, ds, u + i_ds*ds*d2, u_ds + i_ds*d2_ds);

    /* Perform Inplace Transpose To Return To Column Major Format */
    mkl_dimatcopy('C', 'T', d2, d1, 1.0, u, d2, d1);
    mkl_dimatcopy('C', 'T', d2_ds, d1_ds, 1.0, u_ds, d2_ds, d1_ds);

    /* Finish By Interpolating Each Column */
    int j;
    for (j = 0; j < d2; j++) 
        upsample_1d_inplace(d1, ds, u + j*d1);
}
