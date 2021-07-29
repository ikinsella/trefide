#include "welch.h"

#include <vector>


// Explicit Template Specialization To Map dtype -> Config Enums

template<> inline DFTI_CONFIG_VALUE dfti_precision<float>(){ return DFTI_SINGLE; }

template<> inline DFTI_CONFIG_VALUE dfti_precision<double>(){ return DFTI_DOUBLE; }


// Explicit Template Specialization To Generalize MKL Functions 

template<> inline 
void cblas_scal<float>(const MKL_INT n, const float a, float *x, const MKL_INT incx){
    cblas_sscal(n, a, x, incx);
}

template<> inline 
void cblas_scal<double>(const MKL_INT n, const double a, double *x, const MKL_INT incx){
    cblas_dscal(n, a, x, incx);
}

template <> inline 
float cblas_nrm2<float>(const MKL_INT n, const float *x, const MKL_INT incx){
    return cblas_snrm2(n, x, incx); 
}

template <> inline 
double cblas_nrm2<double>(const MKL_INT n, const double *x, const MKL_INT incx){
    return cblas_dnrm2(n, x, incx); 
}

template <> inline
void vMul<float>(const MKL_INT n, const float *a, const float *b, float *y){
    vsMul(n, a, b, y);
}

template <> inline
void vMul<double>(const MKL_INT n, const double *a, const double *b, double *y){
    vdMul(n, a, b, y);
}

template <> inline 
void vSqr<float>(const MKL_INT n, const float *a, float *y){
    vsSqr(n, a, y);
}

template <> inline 
void vSqr<double>(const MKL_INT n, const double *a, double *y){
    vdSqr(n, a, y);
}


// Welch-Specific Functions
template <typename T>
inline void hanning_window(const MKL_INT L, T* win) {
    T rad_inc;

    // Fill Vector With Window Val
    rad_inc = 2 * M_PI / (L - 1);
    for (int l = 0; l < L; l++) {
        win[l] = .5 * (1 - cos(rad_inc * l));
    }
}


template <typename T>
void inplace_rfft(const MKL_LONG L, T *yft, void *FFT) {
    // Declare Local Variables
    MKL_LONG status;
    DFTI_DESCRIPTOR_HANDLE *fft_pt;
    DFTI_DESCRIPTOR_HANDLE fft;
    bool destroy_handle = false;

    // Assign or Create FFT Handle Depending On Context
    if (FFT) {
        /* Void FFT handle indicates operating in serial mode */
        fft_pt = static_cast<DFTI_DESCRIPTOR_HANDLE *>(FFT);
        fft = *fft_pt;
    } else {
        status = DftiCreateDescriptor(&fft, dfti_precision<T>(), DFTI_REAL, 1, L);
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


template <typename T>
void welch(const size_t N, const MKL_INT L, const MKL_INT R, const T fs,
           const T *x, T *psd, void *FFT) {
    int k, l;
    MKL_INT K, S, P;
    T scale;

    MKL_INT temp = std::min(static_cast<MKL_INT>(N), L);
    S = L - R;                                        // Segment Increment
    K = static_cast<MKL_INT>(std::max(floor((static_cast<MKL_INT>(N) - L) / S), 0.0) + 1); // Number Of Segments
    P = static_cast<MKL_INT>(floor(L / 2) + 1); // Number of Periodogram Coef
    std::vector<T> win(L, 0.0);            // Window
    std::vector<T> yft(L, 0.0); // Windowed Signal Segment & DFT Coef

    /* Construct Window & Compute DFT Coef Scaling Factor */
    hanning_window<T>(temp, &win[0]);
    scale = cblas_nrm2<T>(temp, &win[0], 1); // Window 2 norm
    scale = fs * scale * scale;         // DFT Normalization Factor

    // Loop Over Segments
    for (k = 0; k < K; k++) {

        // Copy Segment Of Signal X Into Target YFT & Apply Window
        vMul<T>(temp, x + S * k, &win[0], &yft[0]);

        // Transform Windowed Signal Segment Into Squared DFT Coef. Inplace
        inplace_rfft<T>(L, &yft[0], FFT);
        vSqr<T>(L, &yft[0], &yft[0]);

        // Increment Estimate Of PSD
        psd[0] += yft[0];                 /* DC component */
        for (l = 1; l < ceil(L / 2); l++) /* (l < ceil(L/2) */
            psd[l] += yft[l * 2] + yft[l * 2 - 1];
        if (L % 2 == 0)               /* L is even */
            psd[P - 1] += yft[L - 1]; /* Nyquist freq. */
    }

    // Scale Non-offset and Nyquist Terms By 2
    cblas_scal<T>(P - 2, 2, psd + 1, 1);

    // Scale Periodogram By Win, Fs, and Num Segments
    cblas_scal<T>(P, 1 / (scale * K), psd, 1);
}


template <typename T>
T psd_noise_estimate(const size_t N, const T *x, void *FFT) {
    T var = 0;

    /* Call Pwelch PSD Estimator with  Reasonable Defaults */
    MKL_INT L = 256; // Segment Length
    MKL_INT R = 128; // Segment Overlap
    T fs = 1;   // Sampling Freq Of Signal
    MKL_INT P = static_cast<MKL_INT>(floor(L / 2) + 1); // Num Periodogram Coef
    std::vector<T> psd(P, 0.0);

    welch<T>(N, L, R, fs, x, &psd[0], FFT);

    /* Average Over High Frequency Components */
    int start = static_cast<int>(floor(L / 4) + 1);
    int end = static_cast<int>(floor(L / 2) + 1);
    for (int p = start; p < end; p++) {
        var += psd[p] * 0.5;
    }
    var /= floor(L / 2) - floor(L / 4);

    return var;
}


// Tell Compiler To Instantiate Templates For Each Supported Input Type 
template void inplace_rfft(const MKL_LONG L, float *yft, void *FFT); 
template void inplace_rfft(const MKL_LONG L, double *yft, void *FFT); 
template double psd_noise_estimate(const size_t N, const double *x, void *FFT);
template void welch(const size_t N, const MKL_INT L, const MKL_INT R, const double fs, const double *x, double *psd, void *FFT);
