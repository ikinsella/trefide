#include <stdlib.h>
#include <mkl_dfti.h>
#ifndef WELCH_H
#define WELCH_H

//#ifdef  __cplusplus
//extern "C" {
//#endif

void inplace_rfft(const MKL_LONG L, double* yft, void* FFT=NULL);
void hanning_window(const MKL_INT L, double* win);
void welch(const size_t N, const MKL_INT L, const MKL_INT R, const double fs, 
              const double*x, double* psd, void* FFT=NULL);
double psd_noise_estimate(const size_t N, const double* x, void* FFT=NULL);

//#ifdef  __cplusplus
//}
//#endif
#endif /* WELCH_H */
