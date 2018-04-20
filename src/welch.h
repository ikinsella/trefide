#include <stdlib.h>
#include <mkl_dfti.h>
#ifndef WELCH_H
#define WELCH_H

//#ifdef  __cplusplus
//extern "C" {
//#endif

double* inplace_rfft(const size_t L, double* yft);
void threadsafe_inplace_rfft(const MKL_LONG L, double* yft, 
                             DFTI_DESCRIPTOR_HANDLE *FFT);
void hanning_window(const size_t L, double* win);
double* welch(const size_t N, const size_t L, const size_t R, const size_t fs, 
              const double*x, DFTI_DESCRIPTOR_HANDLE *FFT);
double psd_noise_estimate(const size_t N, const double* x, DFTI_DESCRIPTOR_HANDLE *FFT);

//#ifdef  __cplusplus
//}
//#endif
#endif /* WELCH_H */
