#ifndef WELCH_H
#define WELCH_H

#include <iostream>
#include <math.h>
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
#include <mkl.h>
#pragma GCC diagnostic pop

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Inline Wrapper Functions Neccessary To Convert To Templated Code
template <typename T> inline void cblas_scal(const MKL_INT n, const T a, T *x, const MKL_INT incx);
template <typename T> inline T cblas_nrm2(const MKL_INT n, const T *x, const MKL_INT incx);
template <typename T> inline void vMul(const MKL_INT n, const T *a, const T *b, T *y);
template <typename T> inline void vSqr(const MKL_INT n, const T *a, T *y);
template <typename T> inline DFTI_CONFIG_VALUE dfti_precision();

// Welch-Specific Functions 

template <typename T> inline void hanning_window(const MKL_INT L, T* win);

template <typename T> void inplace_rfft(const MKL_LONG L, T* yft, void* FFT); 

template <typename T> void welch(const size_t N, const MKL_INT L, const MKL_INT R, 
    const T fs, const T *x, T *psd, void *FFT = NULL);

template <typename T> T psd_noise_estimate(const size_t N, const T *x, void *FFT = NULL);

#endif /* WELCH_H */
