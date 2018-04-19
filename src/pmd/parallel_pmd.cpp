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
         
        // Loop Over All Patches In Parallel
        #pragma omp for
        for (m = 0; m < b; m++){
 
            //Use dummy vars for decomposition
            Kpt[m] = factor_patch(bheight, 
                                  bwidth, 
                                  t, 
                                  Rpt[m], 
                                  Upt[m], 
                                  Vpt[m], 
                                  lambda_tv, 
                                  spatial_thresh, 
                                  max_components,
                                  max_iters, 
                                  tol);
            
        } // End Parallel For Loop

    } // End Parallel Section

}

