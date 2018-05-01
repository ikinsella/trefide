# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False

import os

import numpy as np
cimport numpy as np


# -----------------------------------------------------------------------------#
# ------------------------- Imports From Libtrefide.so ------------------------#
# -----------------------------------------------------------------------------#


cdef extern from "trefide.h":

    void downsample_1d(const int t, 
                       const int ds, 
                       const double* v, 
                       double* v_ds) nogil


    void downsample_2d(const int d1, 
                       const int d2, 
                       const int ds, 
                       const double* u, 
                       double* u_ds) nogil


    void upsample_1d_inplace(const int t, 
                             const int ds, 
                             double* v) nogil


    void upsample_1d(const int t, 
                     const int ds, 
                     double* v, 
                     const double* v_ds) nogil

    void upsample_2d(const int d1, 
                     const int d2, 
                     const int ds, 
                     double* u, 
                     double* u_ds) nogil


# -----------------------------------------------------------------------------#
# ---------------------------- Downsampling Wrappers --------------------------#
# -----------------------------------------------------------------------------#


cpdef double[::1] downsample_signal(const int t, 
                                      const int ds,
                                      double[::1] V):
    """ Downsample Image In Each Dimension By Factor Of ds """

    # Assert Dimensions Match
    assert t % ds == 0, "Signal length must be divisible by downsampling factor."

    # Declare & Intiialize Local Variables
    cdef int t_ds = t / ds

    # Allocate Space For Downsampled Singal
    cdef double[::1] V_ds = np.zeros(t_ds, dtype=np.float64)

    # Call C-Routines From Trefide
    with nogil:
        downsample_1d(t, ds, &V[0], &V_ds[0])

    return np.asarray(V_ds)


cpdef double[::1,:] downsample_image(const int d1, 
                                     const int d2, 
                                     const int ds,
                                     double[::1] U):
    """ Downsample Image In Each Dimension By Factor Of ds """

    # Assert Dimensions Match
    assert d1 % ds == 0, "Height of image must be divisible by downsampling factor."
    assert d2 % ds == 0, "Width of image must be divisible by downsampling factor."

    # Declare & Intiialize Local Variables
    cdef int d1_ds = d1 / ds
    cdef int d2_ds = d2 / ds

    # Allocate Space For Downsampled Image
    cdef double[::1] U_ds = np.zeros(d1_ds * d2_ds, dtype=np.float64)

    # Call C-Routines From Trefide
    with nogil:
        downsample_2d(d1, d2, ds, &U[0], &U_ds[0])

    return np.reshape(U_ds, (d1_ds, d2_ds), order='F')


# -----------------------------------------------------------------------------#
# ---------------------------- Upsampling Wrappers --------------------------#
# -----------------------------------------------------------------------------#

cpdef double[::1,:] upsample_image(const int d1, 
                                   const int d2, 
                                   const int ds,
                                   double[::1] U_ds):
    """ Downsample Image In Each Dimension By Factor Of ds """

    # Assert Dimensions Match
    assert d1 % ds == 0, "Height of image must be divisible by downsampling factor."
    assert d2 % ds == 0, "Width of image must be divisible by downsampling factor."
    
    # Allocate Space For Upsampled Image
    cdef double[::1] U = np.zeros(d1 * d2, dtype=np.float64)

    # Call C-Routines From Trefide
    with nogil:
        upsample_2d(d1, d2, ds, &U[0], &U_ds[0])

    return np.reshape(U, (d1, d2), order='F')


cpdef double[::1] upsample_signal(const int t, 
                                  const int ds,
                                  double[::1] V_ds):
    """ Downsample Image In Each Dimension By Factor Of ds """

    # Assert Dimensions Match
    assert t % ds == 0, "Signal length must be divisible by downsampling factor."

    # Allocate Space For Upsampled Singal
    cdef double[::1] V = np.zeros(t, dtype=np.float64)

    # Call C-Routines From Trefide
    with nogil:
        upsample_1d(t, ds, &V[0], &V_ds[0])

    return np.asarray(V)
