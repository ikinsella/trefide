#include <stdlib.h>
#ifndef WELCH_H
#define WELCH_H

#ifdef  __cplusplus
extern "C" {
#endif

double* inplace_rfft(const size_t L, double* yft);
void hanning_window(const size_t L, double* win);
double* welch(const size_t N, const size_t L, const size_t R, const size_t fs, const double*x);
double psd_noise_estimate(const size_t N, const double* x);

#ifdef  __cplusplus
}
#endif
#endif /* WELCH_H */
