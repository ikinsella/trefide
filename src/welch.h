#ifndef WELCH_H
#define WELCH_H

#include <math.h>
#include <mkl_dfti.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

void inplace_rfft(const MKL_LONG L, double* yft, void* FFT = NULL);

inline void hanning_window(const MKL_INT L, double* win)
{
    double rad_inc;

    // Fill Vector With Window Val
    rad_inc = 2 * M_PI / (L - 1);
    for (int l = 0; l < L; l++) {
        win[l] = .5 * (1 - cos(rad_inc * l));
    }
}

void welch(const size_t N, const MKL_INT L, const MKL_INT R, const double fs,
    const double* x, double* psd, void* FFT = NULL);

double psd_noise_estimate(const size_t N, const double* x, void* FFT = NULL);

#endif /* WELCH_H */
