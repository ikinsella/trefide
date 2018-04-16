#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mkl_dfti.h>
#include <mkl.h>
#ifndef M_PI
#define M_PI           3.14159265358979323846
#endif


double* inplace_rfft(const size_t L, double* yft){
    
    // Declare Local Variables
    DFTI_DESCRIPTOR_HANDLE desc_handle;
    MKL_LONG status;

    /* result is x_out[0], ..., x_out[31]*/
    status = DftiCreateDescriptor( &desc_handle, DFTI_DOUBLE, DFTI_REAL, 1, L );
    status = DftiSetValue( desc_handle, DFTI_PACKED_FORMAT, DFTI_PACK_FORMAT);
    status = DftiCommitDescriptor( desc_handle );
    status = DftiComputeForward( desc_handle, yft );
    status = DftiFreeDescriptor( &desc_handle );
 
    /* Check Status & Warn */
    if (status != 0) fprintf(stderr, "MKL_FFT: %ld\n", status);

    return yft;
}


void hanning_window(const size_t L, double* win){
   
   // Declare Internal Vars 
   size_t l;
   double rad_inc;

   // Fill Vector With Window Val
   rad_inc = 2 * M_PI / (L - 1);
   for (l = 0; l < L; l++){
       win[l] = .5 * (1 - cos(rad_inc * l));
   }
}


double* welch(const size_t N, const size_t L, const size_t R, const size_t fs, const double*x){
    
    /* Declare Local Variables */
    size_t k, l, K, S, P;
    double *yft, *win, *psd;
    double scale;

    /* Initialize & Allocate Mem For Local Variables */
    S = L - R;  // Segment Increment 
    K = floor((N - L) / S);  // Number Of Segments
    P = floor(L / 2) + 1;  // Number of Periodogram Coef
    win = (double *)malloc(L*sizeof(double));  // Window
    yft = (double *)malloc(L*sizeof(double));  // Windowed Signal Segment & DFT Coef
    psd = (double *)malloc(P*sizeof(double));  // Averaged Periodogram Output 

    /* Construct Window & Compute DFT Coef Scaling Factor */
    hanning_window(L, win);
    scale = cblas_dnrm2(L, win, 1);  // Window 2 norm
    scale = fs * scale * scale;  // DFT Normalization Factor

    // Loop Over Segments
    for (k=0; k < K; k++){

        // Copy Segment Of Signal X Into Target YFT & Apply Window
        vdMul(L, x + S * k, win, yft);
        
        // Transform Windowed Signal Segment Into Squared DFT Coef. Inplace
        inplace_rfft(L, yft);
        vdSqr(L, yft, yft);

        // Increment Estimate Of PSD
        psd[0] += yft[0];  /* DC component */
        for (l = 1; l < ceil(L/2); l++)  /* (l < ceil(L/2) */
            psd[l] += yft[l*2] + yft[l*2-1];
        if (L % 2 == 0) /* L is even */
            psd[P-1] += yft[L-1];  /* Nyquist freq. */
    }

    // Scale Non-offset and Nyquist Terms By 2
    cblas_dscal(P - 2, 2, psd + 1, 1);

    // Scale Periodogram By Win, Fs, and Num Segments
    cblas_dscal(P, 1 / (scale * K), psd, 1);

    // Free Allocated Memory
    free(yft);
    free(win);

    return psd; 
}


double psd_noise_estimate(const size_t N, const double* x){

    /* Declare Internal Vars */
    double *psd;
    double var;
    size_t p;

    /* Call Pwelch PSD Estimator with  Reasonable Defaults */
    size_t L = 256; // Segment Length
    size_t R = 128; // Segment Overlap
    size_t fs = 1;  // Sampling Freq Of Signal
    psd = welch(N, L, R, fs, x);
    
    /* Average Over High Frequency Components */
    for (p=floor(L / 4) + 1; p < floor(L / 2) + 1; p++){
       var += psd[p] / 2; 
    }
    var /= floor(L / 2) - floor(L / 4);

    /* Free Memory Allocated By Welch */
    free(psd);
    return var;
}

