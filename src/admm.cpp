// TODO: figure out why changing the ordering causes issues
#include "line_search.h" /* compute_scale */
#include "glmgen.h"
#include "admm.h"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
#include <mkl.h>
#pragma GCC diagnostic pop

#include <math.h>
#include <vector>

short constrained_tf_admm(const int n, double *x, double *y, double *w,
                          const double delta, double *beta, double *alpha,
                          double *u, double *lambda, int *iters,
                          const double tol, const int verbose) {
    /* Trend Filtering Constants */
    int DEGREE = 1;

    /* Declare & Initialize Local Variables */
    short status;
    int i;
    double scale, rho = 1;
    std::vector<double> temp_n(n);
    for (i = 0; i < n; i++) {
        temp_n[i] = 1.0 / sqrt(w[i]); // Assume w does not have zeros
    }

    /* Declare & Init Sparse Matrix Objects */
    cs *D = tf_calc_dk(n, DEGREE + 1, x);
    cs *Dt = cs_transpose(D, 1);
    diag_times_sparse(Dt, &temp_n[0]); /* Dt = W^{-1/2} Dt */
    cs *Dk = tf_calc_dktil(n, DEGREE, x);
    cs *Dkt = cs_transpose(Dk, 1);
    cs *DktDk = cs_multiply(Dkt, Dk);
    gqr *Dt_qr = glmgen_qr(Dt);
    gqr *Dkt_qr = glmgen_qr(Dkt);

    /* If Uninitialized, Compute Starting Points For Search */
    if (*lambda <= 0) {

        /* Compute Step Size wrt Transformed Lambda Space*/
        scale = compute_scale(n, y, delta);
        *lambda = exp((log(20 + (1 / scale)) - log(3 + (1 / scale))) / 2 +
                      log(3 * scale + 1)) -
                  1;
    }

    /* Compute Rho From Data Locations */
    rho *= pow((x[n - 1] - x[0]) / n, static_cast<double>(DEGREE));

    /* if lambda is too small, return a trivial solution */
    if (*lambda <= 1e-10 * l1norm(y, n) / n) {
        for (i = 0; i < n; i++) {
            beta[i] = y[i];
        }

        *lambda = 0;
        cs_spfree(D);
        cs_spfree(Dt);
        cs_spfree(Dk);
        cs_spfree(Dkt);
        cs_spfree(DktDk);
        glmgen_gqr_free(Dt_qr);
        glmgen_gqr_free(Dkt_qr);
        return 1;
    }

    /* v_k <- argmin_{v_k} ||v_k||_TF s.t. ||v - v_k||_2^2 <= T * delta */
    status = cps_tf_admm(n, DEGREE, x, y, w, DktDk, delta, beta, alpha, u,
                         lambda, rho, iters, tol, verbose);

    /* Free Allocated Memory */
    cs_spfree(D);
    cs_spfree(Dt);
    cs_spfree(Dk);
    cs_spfree(Dkt);
    cs_spfree(DktDk);
    glmgen_gqr_free(Dt_qr);
    glmgen_gqr_free(Dkt_qr);

    return status;
}

short cps_tf_admm(const int n, const int degree, double *x, double *y,
                  double *w, cs *DktDk, const double delta, double *beta,
                  double *alpha, double *u, double *lambda, double rho,
                  int *iters, const double tol, const int verbose) {
    /* TF constants */
    int maxiter = 100;
    double obj_tol = 1e-4;

    /* Declare & allocate internal admm vars */
    int df;
    std::vector<double> obj(maxiter);

    /* Declar & allocate internal line search vars */
    double target = sqrt(n * delta); // target norm of error
    int iter;
    double l2_err, l2_err_prev = 0;
    std::vector<double> resid(n);

    /* Iterate lagrangian solves over lambda; the beta, alpha, and u vectors
     * get used for warm starts in each subsequent iteration.
     */
    while (*iters < maxiter * 100) {
        /* fit admm */
        tf_admm_gauss(x, y, w, n, degree, maxiter, *lambda, &df, beta, alpha, u,
                      &obj[0], &iter, rho * (*lambda), obj_tol, DktDk, verbose);

        /* If there any NaNs in beta: reset beta, alpha, u */
        if (has_nan(beta, n)) {
            return -1; // abort search
        }

        /* Increment LS Total Iters */
        *iters += iter;

        /* Compute norm of residual */
        vdSub(n, y, beta, &resid[0]);
        l2_err = cblas_dnrm2(n, &resid[0], 1);

        /* Check For Convergence */
        if (fabs(target * target - l2_err * l2_err) / (target * target) < tol) {
            return 1; // successfully converged within tolerance
        } else if (fabs(l2_err - l2_err_prev) < 1e-3) {
            return 0; // Stalled before tol reached
        }

        l2_err_prev = l2_err;

        /* Increment Lambda */
        // *lambda = exp(log(*lambda) + log(target) - log(l2_err));
        *lambda = (*lambda) * target / l2_err;
    }

    /* Line Search Didn't Converge */
    return 0; // Reached Maxiter before tol reached
}

short langrangian_tf_admm(const int n, double *x, double *y, double *w,
                          double lambda, double *beta, double *alpha, double *u,
                          int *iter, const int verbose) {
    /* Trend Filtering Constants */
    int DEGREE = 1;
    int maxiter = 100;
    double obj_tol = 1e-4;

    /* Declare & allocate internal admm vars */
    int i, df;
    short status = 1;
    double rho = 1;
    std::vector<double> obj(maxiter);
    std::vector<double> temp_n(n);
    for (i = 0; i < n; i++) {
        temp_n[i] = 1.0 / sqrt(w[i]); // Assume w does not have zeros
    }

    /* Declare & Init Sparse Matrix Objects */
    cs *D = tf_calc_dk(n, DEGREE + 1, x);
    cs *Dt = cs_transpose(D, 1);
    diag_times_sparse(Dt, &temp_n[0]); /* Dt = W^{-1/2} Dt */
    cs *Dk = tf_calc_dktil(n, DEGREE, x);
    cs *Dkt = cs_transpose(Dk, 1);
    cs *DktDk = cs_multiply(Dkt, Dk);
    gqr *Dt_qr = glmgen_qr(Dt);
    gqr *Dkt_qr = glmgen_qr(Dkt);

    /* Compute Rho From Data Locations */
    rho *= pow((x[n - 1] - x[0]) / n, static_cast<double>(DEGREE));

    /* if lambda is too small, return a trivial solution */
    if (lambda <= 1e-10 * l1norm(y, n) / n) {

        for (i = 0; i < n; i++) {
            beta[i] = y[i];
        }
        lambda = 0;

        cs_spfree(D);
        cs_spfree(Dt);
        cs_spfree(Dk);
        cs_spfree(Dkt);
        cs_spfree(DktDk);
        glmgen_gqr_free(Dt_qr);
        glmgen_gqr_free(Dkt_qr);
        return status;
    }

    /* fit admm */
    tf_admm_gauss(x, y, w, n, DEGREE, maxiter, lambda, &df, beta, alpha, u,
                  &obj[0], iter, rho * lambda, obj_tol, DktDk, verbose);

    /* If there any NaNs in beta: solve failed */
    if (has_nan(beta, n)) {
        status = -1;
    }

    /* Free Allocated Memory */
    cs_spfree(D);
    cs_spfree(Dt);
    cs_spfree(Dk);
    cs_spfree(Dkt);
    cs_spfree(DktDk);
    glmgen_gqr_free(Dt_qr);
    glmgen_gqr_free(Dkt_qr);

    return status;
}
