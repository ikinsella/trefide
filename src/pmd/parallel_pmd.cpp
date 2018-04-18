#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mkl.h>
#include "trefide.h"


void parrallel_factor_patch(const MKL_INT bheight, 
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
                            const double tol)
{
    // Declare Internal Variables
    int m;

    // Start Parallel Region So Each Thread Can Allocate It's Own Memory
    #pragma omp parallel
    {
        //Allocate Memory To Dummy Pointers For Each Thread
        size_t K;
        double* R = (double *) malloc(bheight*bwidth*t*sizeof(double));
        double* U = (double *) malloc(bheight*bwidth*max_components*sizeof(double));
        double* V = (double *) malloc(t*max_components*sizeof(double));
         

        // Loop Over All Patches In Parallel
        #pragma omp for
        for (m = 0; m < b; m++){

            // Copy Block Data To Residual Dummy Pointer
            copy(bheight*bwidth*t, Rpt[m], R);
            
            //Use dummy vars for decomposition
            K = factor_patch(bheight, 
                             bwidth, 
                             t, 
                             R, 
                             U, 
                             V, 
                             lambda_tv, 
                             spatial_thresh, 
                             max_components,
                             max_iters, 
                             tol);
            
            //Copy Components Back From Dummy Pointers
            copy(bheight*bwidth*max_components, U, Upt[m]);
            copy(t*max_components, V, Vpt[m]);
            Kpt[m] = K;

        } // End Parallel For Loop
    
        //Free Memory Allocated Within Each thread
        free(R);
        free(U);
        free(V);

    } // End Parallel Section

}

