/* l1tf.h
 *
 * Copyright (C) 2007 Kwangmoo Koh, Seung-Jean Kim and Stephen Boyd.
 *
 * See the file "COPYING.TXT" for copyright and warranty information.
 *
 * Author: Kwangmoo Koh (deneb1@stanford.edu)
 */
#ifndef IPM_H
#define IPM_H

/* main routine for l1 trend filtering */
int l1tf(const int n, const double *y, const double lambda, double *x,
         double *z, int *iter, const double tol, const int maxiter,
         const int verbose);

/* utilit to compte the maximum value of lambda */
double l1tf_lambdamax(const int n, double *y, const int verbose);

void yainvx(int n, const double a, const double *x, double *y); /* y = a./x */
void vecset(int n, const double val, double *dst);
double vecsum(int n, const double *src);

void print_ivec(int n, const int *x); /* for debug */

#endif /* IPM_H */
