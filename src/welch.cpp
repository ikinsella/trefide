#include "welch.h"
#include <mkl.h>
#include <mkl_dfti.h>
#include <stdio.h>
#include <vector>

void inplace_rfft(const MKL_LONG L, double* yft, void* FFT)
{
    // Declare Local Variables
    MKL_LONG status;
    DFTI_DESCRIPTOR_HANDLE* fft_pt;
    DFTI_DESCRIPTOR_HANDLE fft;
    bool destroy_handle = false;

    // Assign or Create FFT Handle Depending On Context
    if (FFT) {
        /* Void FFT handle indicates operating in serial mode */
        fft_pt = (DFTI_DESCRIPTOR_HANDLE*)FFT;
        fft = *fft_pt;
    } else {
        status = DftiCreateDescriptor(&fft, DFTI_DOUBLE, DFTI_REAL, 1, L);
        if (status != 0)
            fprintf(stderr, "Error Creating FFT Handle: %ld\n", status);
        destroy_handle = true;
        status = DftiSetValue(fft, DFTI_PACKED_FORMAT, DFTI_PACK_FORMAT);
        status = DftiCommitDescriptor(fft);
    }

    // Compute Forward FFT Modifying Input Inplace
    status = DftiComputeForward(fft, yft);
    if (status != 0)
        fprintf(stderr, "Error Computing Forward FFT: %ld\n", status);

    // Destroy FFT Handle If Created Internally
    if (destroy_handle) {
        status = DftiFreeDescriptor(&fft);
        if (status != 0)
            fprintf(stderr, "Error Destroying FFT Hanlde: %ld\n", status);
    }
}

void welch(const size_t N, const MKL_INT L, const MKL_INT R,
    const double fs, const double* x, double* psd, void* FFT)
{
    /* Declare Local Variables */
    int k, l;
    MKL_INT K, S, P;
    double *yft, *win;
    double scale;

    /* Initialize & Allocate Mem For Local Variables */
    S = L - R; // Segment Increment
    K = floor((N - L) / S) + 1; // Number Of Segments
    P = floor(L / 2) + 1; // Number of Periodogram Coef
    win = (double*)malloc(L * sizeof(double)); // Window
    yft = (double*)malloc(L * sizeof(double)); // Windowed Signal Segment & DFT Coef

    /* Construct Window & Compute DFT Coef Scaling Factor */
    hanning_window(L, win);
    scale = cblas_dnrm2(L, win, 1); // Window 2 norm
    scale = fs * scale * scale; // DFT Normalization Factor

    // Loop Over Segments
    for (k = 0; k < K; k++) {

        // Copy Segment Of Signal X Into Target YFT & Apply Window
        vdMul(L, x + S * k, win, yft);

        // Transform Windowed Signal Segment Into Squared DFT Coef. Inplace
        inplace_rfft(L, yft, FFT);
        vdSqr(L, yft, yft);

        // Increment Estimate Of PSD
        psd[0] += yft[0]; /* DC component */
        for (l = 1; l < ceil(L / 2); l++) /* (l < ceil(L/2) */
            psd[l] += yft[l * 2] + yft[l * 2 - 1];
        if (L % 2 == 0) /* L is even */
            psd[P - 1] += yft[L - 1]; /* Nyquist freq. */
    }

    // Scale Non-offset and Nyquist Terms By 2
    cblas_dscal(P - 2, 2, psd + 1, 1);

    // Scale Periodogram By Win, Fs, and Num Segments
    cblas_dscal(P, 1 / (scale * K), psd, 1);

    // Free Allocated Memory
    free(yft);
    free(win);
}

double psd_noise_estimate(const size_t N, const double* x, void* FFT)
{
    // double* psd;
    double var = 0;
    // int p;

    /* Call Pwelch PSD Estimator with  Reasonable Defaults */
    MKL_INT L = 256; // Segment Length
    MKL_INT R = 128; // Segment Overlap
    double fs = 1; // Sampling Freq Of Signal
    MKL_INT P = floor(L / 2) + 1; // Number of Periodogram Coef
    /*
    psd = (double*)malloc(P * sizeof(double));
    for (int p = 0; p < P; p++) {
        psd[p] = 0.0;
    }
    */
    std::vector<double> psd(P, 0.0);

    welch(N, L, R, fs, x, &psd[0], FFT);

    /* Average Over High Frequency Components */
    for (int p = floor(L / 4) + 1; p < floor(L / 2) + 1; p++) {
        var += psd[p] * 0.5;
    }
    var /= floor(L / 2) - floor(L / 4);

    /* Free Memory Allocated By Welch */
    // free(psd);
    return var;
}
