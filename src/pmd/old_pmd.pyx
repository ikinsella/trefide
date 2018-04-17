# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: nonecheck=False


cimport cython

import numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free
from libc.stdio cimport fprintf, stderr
from trefide.solvers.time.constrained cimport cpdas_lite, compute_scale
#from trefide.utils.welch cimport psd_noise_estimate
#from trefide.solvers.space.lagrangian import ldr2_tv

# -----------------------------------------------------------------------------#
# --------------------------- Generic Helper Funcs ----------------------------#
# -----------------------------------------------------------------------------#


cdef extern from "math.h":
    double sqrt(double m) nogil
    double log(double m) nogil
    double exp(double m) nogil
    double pow(double m, int n) nogil
    double fabs(double m) nogil


cdef extern from "proxtv.h":
    int DR2_TV(size_t M, 
               size_t N, 
               double*unary, 
               double W1, 
               double W2, 
               double norm1, 
               double norm2, 
               double*s, 
               int nThreads, 
               int maxit, 
               double* info) nogil


cdef extern from "trefide.h":
    double psd_noise_estimate (const size_t n,
                               const double *x) nogil


cdef double arr_distance(const size_t n, 
                         const double* x,
                         const double* y) nogil:
    """Compute ||x - y||_2 where x,y are len n arrs"""

    # Declare Internal Variables
    cdef double norm = 0
    cdef size_t i

    # Compute Norm
    for i in range(n):
        norm += pow(x[i] - y[i], 2)
    return sqrt(norm)


cdef void arr_copy(const size_t n, 
                   const double* source,
                   double* dest) nogil:
    """Copy leading n vals of arr source leading n vals of array dest"""

    # Declare Internal Variables
    cdef size_t i

    # Copy Contents
    for i in range(n):
        dest[i] = source[i]


cdef void arr_normalize(const size_t n,
                        double* x) nogil:
    """Normlizes len n array x by ||x||_2 in place"""
    
    # Declare Internal Variables
    cdef size_t i
    cdef double norm = 0
    
    # Compute Vector Norm
    for i in range(n):
        norm += x[i] * x[i]
    norm = sqrt(norm)

    # Normalize Vector
    for i in range(n):
        x[i] /= norm


# -----------------------------------------------------------------------------#
# --------------------------- Spatial Helper Funcs ----------------------------#
# -----------------------------------------------------------------------------#


cdef void regress_spatial(const size_t d1,
                           const size_t d2,
                           const size_t t,
                           const double[:, :, ::1] R_k, 
                           double[::1] u_k, 
                           const double[::1] v_k) nogil:
    """Updates & normalizes the spatial component u_k in place by regressing 
    the current temporal component v_k against the current residual R_k"""

    # Declare Internal Vars
    cdef size_t j = 0
    cdef size_t i, j1, j2
    cdef double norm = 0
    
    # u = Yv, norm = ||Yv||_2
    for j2 in range(d2):
        for j1 in range(d1):
            u_k[j] = 0
            for i in range(t):
                u_k[j] += R_k[j1, j2, i] * v_k[i]
            norm += u_k[j] * u_k[j]
            j+=1
    norm = sqrt(norm)

    # u = Yv / ||Yv||_2 
    for j in range(d1 * d2):
        u_k[j] /= norm


cdef void denoise_spatial(const size_t d1,
                          const size_t d2,
                          double[::1] u_k,
                          const double lambda_tv) nogil:
    """Denoises and normalizes the temporal component v_k using the 
    instantiated trefide TrendFilter object"""

    # Declare Internal Vars
    cdef size_t j
    cdef double* unary = <double *> malloc(d1 * d2 * sizeof(double))
    cdef double* info = <double *> malloc(3 * sizeof(double))

    # Copy Image Into Target
    for j in range(d1*d2):
        unary[j] = u_k[j]

    # Call TV solver To Update u_k inplace
    DR2_TV(d1, d2, unary, lambda_tv, lambda_tv, 1, 1, &u_k[0], 1, 1, info)
    arr_normalize(d1*d2, &u_k[0])

    # Free Allocated Memory
    free(unary);
    free(info);


cdef double update_spatial(const size_t d1,
                                 const size_t d2,
                                 const size_t t,
                                 const double[:, :, ::1] R_k,
                                 double[::1] u_k,
                                 double* u__,
                                 const double[::1] v_k,
                                 const double lambda_tv) nogil:
    """ """
    arr_copy(d1 * d2, &u_k[0], u__)
    regress_spatial(d1, d2, t, R_k, u_k, v_k)
    denoise_spatial(d1, d2, u_k, lambda_tv)
    return arr_distance(d1 * d2, &u_k[0], u__)


# -----------------------------------------------------------------------------#
# --------------------------- Temporal Helper Funcs ---------------------------#
# -----------------------------------------------------------------------------#


cdef void regress_temporal(const size_t d1,
                           const size_t d2,
                           const size_t t,
                           const double[:, :, ::1] R_k, 
                           const double[::1] u_k, 
                           double[::1] v_k) nogil:
    """Updates the temporal component v_k in place by regressing 
    the current spatial component u_k against the current residual R_k"""

    # Declare Internal Vars
    cdef size_t j = 0
    cdef size_t i, j1, j2

    # v = Y'u
    for i in range(t):
        v_k[i] = 0 
        for j2 in range(d2):
            for j1 in range(d1):
                v_k[i] += R_k[j1, j2, i] * u_k[j]
                j += 1
        j=0


cdef void denoise_temporal(const size_t t,
                           const double *wi_k,
                           double[::1] v_k,
                           double *z_k,
                           double *lambda_tf) nogil:
    """Denoises and normalizes the temporal component v_k using the 
    instantiated trefide TrendFilter object"""

    cdef double delta = psd_noise_estimate(t, &v_k[0])
    # Denoise updated temporal component 
    cpdas_lite(t, delta, wi_k, &v_k[0], z_k, lambda_tf, 1)

    # Normalize by 2 norm
    arr_normalize(t, &v_k[0])


cdef double update_temporal(const size_t d1,
                            const size_t d2,
                            const size_t t,
                            const double[:, :, ::1] R_k, 
                            const double[::1] u_k,
                            double[::1] v_k,
                            double *v__,
                            const double *wi_k,
                            double *z_k,
                            double *lambda_tf) nogil:
    """ """
    arr_copy(t, &v_k[0], v__)
    regress_temporal(d1, d2, t, R_k, u_k, v_k)
    denoise_temporal(t, wi_k, v_k, z_k, lambda_tf)
    return arr_distance(t, &v_k[0], v__)


# -----------------------------------------------------------------------------#
# -------------------------- Inner Loop (Seq) Funcs ---------------------------#
# -----------------------------------------------------------------------------#


cdef double spatial_test_statistic(const size_t d1,
                                   const size_t d2,
                                   const double[::1] u_k) nogil:
    """ Computes the ratio of the L1 and TV norms as a test statistic """
    
    # Declare Internal Variables
    cdef size_t j, j1, j2
    cdef double norm_tv = 0
    cdef double norm_l1 = u_k[d1*d2-1]  # Bottom, Right Corner

    # All Elements Except Union (Bottom Row, Rightmost Column) 
    for j2 in range(d2-1):
        for j1 in range(d1-1):
            j = d1 * j2 + j1
            norm_tv += fabs(u_k[j] - u_k[j+1]) + fabs(u_k[j] - u_k[j + d1])
            norm_l1 += fabs(u_k[j])

    # Rightmost Column (Skip Bottom Element)
    j = d1 * (d2-1)
    for j1 in range(d1-1):
        norm_tv += fabs(u_k[j] - u_k[j+1])
        norm_l1 += fabs(u_k[j])
        j += 1

    # Bottom Row (Skip Rightmost Element)
    j = d1 - 1
    for j2 in range(d2-1):
        norm_tv += fabs(u_k[j] - u_k[j + d1])
        norm_l1 += fabs(u_k[j])
        j += d1

    # Return Test Statistic
    return norm_l1 / norm_tv


cdef double initialize_components(const size_t d1,
                                  const size_t d2,
                                  const size_t t,
                                  const double[:, :, ::1] R_k,
                                  double[::1] u_k,
                                  double[::1] v_k,
                                  double *wi_k,
                                  double *z_k) nogil: 
    """ Intialize with equivalent of temporal upate where u_k =1/sqrt(d) """
    
    # Declare Internal Variables
    cdef size_t i
    cdef double scale, lambda_tf, delta

    # Initialize Spatial To Constant Vector
    u_k[:] = 1 / sqrt(d1 * d2)

    # Regress spatial vector against residual
    regress_temporal(d1, d2, t, R_k, u_k, v_k)

    # Intialize TF Weights & Dual Warm Start Var
    for i in range(t-2):
        z_k[i] = 0
        wi_k[i] = 1
    wi_k[t-2] = 1
    wi_k[t-1] = 1

    # Compute Initial Guess Of Lambda
    delta = psd_noise_estimate(t, &v_k[0])
    fprintf(stderr, "delta: %1.2e", delta)
    scale = compute_scale(t, &v_k[0], delta)
    fprintf(stderr, "scale: %1.2e", scale)
    lambda_tf = exp((log(20+(1/scale)) - log(3+(1/scale))) / 2 + log(3*scale + 1)) - 1
    fprintf(stderr, "lambda: %1.2e", lambda_tf)
    # Denoise intialized temporal component 
    denoise_temporal(t, wi_k, v_k, z_k, &lambda_tf)

    return lambda_tf


cdef int rank_one_decomposition(const size_t d1, 
                                 const size_t d2, 
                                 const size_t t,
                                 const double[:, :, ::1] R_k, 
                                 double[::1] u_k, # Fortran Order 2D Arr
                                 double[::1] v_k,
                                 const double lambda_tv,
                                 const double spatial_thresh,
                                 const size_t max_iters,
                                 const double tol) nogil:
    """ Solve: 
        u_k, v_k = min <u_k, R_k v_k> - lambda_tv ||u_k||_TV - lambda_tf ||v_k||_TF
    """
 
    # Declare & Allocate Mem For Internal Vars
    cdef size_t iters = 0
    cdef double delta_u, delta_v, lambda_tf
    cdef double *u__ = <double *> malloc(d1 * d2 * sizeof(double))
    cdef double *v__ = <double *> malloc(t * sizeof(double))
    cdef double *wi_k = <double *> malloc(t * sizeof(double))
    cdef double *z_k = <double *> malloc((t-2) * sizeof(double))

    # Initialize Components
    lambda_tf = initialize_components(d1, d2, t, R_k, u_k, v_k, wi_k, z_k)

    # Loop Until Convergence
    while iters < max_iters:

        # Update Components
        delta_u = update_spatial(d1, d2, t, R_k, u_k, u__, v_k, lambda_tv)
        delta_v = update_temporal(d1, d2, t, R_k, u_k, v_k, v__, wi_k, z_k, &lambda_tf)

        # Check Convergence
        if max(delta_u, delta_v) < tol:    

            # Free Allocated Mem
            free(u__)
            free(v__)
            free(wi_k)
            free(z_k)

            # Test Spatial Component Against Null
            if spatial_test_statistic(d1, d2, u_k) < spatial_thresh:
                
                # Discard Component
                return -1

            # Keep Component
            return 1

        # Early Check To See If We're Fitting Noise
        if iters == 9:
            if spatial_test_statistic(d1, d2, u_k) < spatial_thresh:
                
                # Free Allocated Mem
                free(u__)
                free(v__)
                free(wi_k)
                free(z_k)
                
                # Discard Component
                return -1 
        
        # Continue Updating Components
        iters += 1 

    # Convergence Not Met In Maxiter Iterations
    
    # Free Allocated Mem
    free(u__)
    free(v__)
    free(wi_k)
    free(z_k)
   
    # Test Spatial Component Against Null
    if spatial_test_statistic(d1, d2, u_k) < spatial_thresh:
        
        # Discard Component
        return -1
    
    # Keep Component
    return 1


cdef void update_residual(const size_t d1,
                           const size_t d2,
                           const size_t t,
                           double[:,:,::1] R_k,
                           const double[::1] u_k,
                           const double[::1] v_k) nogil:
    """ """
    
    # Declare Internal Variables
    cdef size_t j = 0
    cdef size_t i, j1, j2

    # Loop Over Residual And Remove Rank 1 Component
    for j2 in range(d2):
        for j1 in range(d1):
            for i in range(t):
                R_k[j1, j2, i] -= u_k[j] * v_k[i]
            j += 1


cpdef size_t factor_patch(const size_t d1, 
                             const size_t d2, 
                             const size_t t,
                             double[:, :, ::1] Y, 
                             double[:, ::1] U,
                             double[:, ::1] V,
                             const double lambda_tv,
                             const double spatial_thresh,
                             const size_t max_components,
                             const size_t max_iters,
                             const double tol) nogil:
    """ 
    Iteratively factor patch into spatial and temporal components 
    with a penalized matrix decomposition
    """
 
    # Declare Internal Vars
    cdef int s
    cdef size_t k = 0

    # Loop Until Convergence
    while k < max_components:

        # Extract Top Singular Vectors & Value
        s = rank_one_decomposition(d1, 
                                    d2, 
                                    t, 
                                    Y, 
                                    U[k, :], 
                                    V[k, :], 
                                    lambda_tv, 
                                    spatial_thresh, 
                                    max_iters, 
                                    tol)
        # Check Convergence
        if s < 0:
            return k 
        
        # Debias Temporal & Update Residual
        regress_temporal(d1, d2, t, Y, U[k, :], V[k, :])
        update_residual(d1, d2, t, Y, U[k, :], V[k, :])
        k += 1 

    # Convergergence Not Met After Extracting Max_Components
    return k 

# -----------------------------------------------------------------------------#
# -------------------------- Outer Loop (Par) Funcs ---------------------------#
# -----------------------------------------------------------------------------#


