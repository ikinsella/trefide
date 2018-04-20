#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mkl.h>
#include <mkl_dfti.h>
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
    // Create FFT Handle So It can Be Chared Aross Threads
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
        Kpt[m] = threadsafe_factor_patch(bheight, 
                                         bwidth, 
                                         t, 
                                         Rpt[m], 
                                         Upt[m], 
                                         Vpt[m], 
                                         lambda_tv, 
                                         spatial_thresh, 
                                         max_components,
                                         max_iters, 
                                         tol,
                                         &FFT);
    } // End Parallel For Loop

    // Free MKL FFT Handle
    status = DftiFreeDescriptor( &FFT ); 
    if (status != 0)
        fprintf(stderr, "Error while deallocating MKL_FFT Handle: %ld\n", status);
}
