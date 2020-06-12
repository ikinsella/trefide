#include "wpdas.h"
#include "utils.h"
#include <algorithm>
#include <iostream>
#include <math.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wredundant-decls"
#include <mkl.h>
#pragma GCC diagnostic pop

#include <vector>

/******************************************************************************
 *                                   Globals                                  *
 ******************************************************************************/

/* Macro Function Definitions */
#define max(x, y) ((x) > (y) ? (x) : (y))
#define min(x, y) ((x) < (y) ? (x) : (y))

/******************************************************************************
 *                                 Main Solver                                *
 ******************************************************************************/

short weighted_pdas(const int n, const double* y, const double* wi, const
        double lambda, double* x, double* z, int* iter, double p, const int m,
        const double delta_s, const double delta_e, const int maxiter, const
        int verbose)
{

    std::vector<double> diff_x((n - 2));
    std::vector<double> div_zi(n);
    std::vector<double> vio_fitness((n - 2));
    std::vector<int> vio_index((n - 2));
    std::vector<int> vio_queue(m, n);
    std::vector<int> vio_sort((n - 2));
    int queue_index;
    int min_queue;
    int min_queue_index;
    int max_queue;
    int max_queue_index;
    std::vector<double> ab((n * 3));
    std::vector<double> b(n);
    int n_vio;
    int n_active;

    // Prepare Queue Variables
    queue_index = 0;
    min_queue = n;
    min_queue_index = 0;
    max_queue = n;
    max_queue_index = m - 1;

    /* prepare to begin optimization */
    if (verbose) {
        std::cerr << "____________________________" << std::endl;
        std::cerr << "|Iter|Violators|Active|Prop|" << std::endl;
    }

    // Opt Routine Main Loop
    for (int local_iter = 0; local_iter < maxiter; local_iter++) {

        /* Subspace Minimization */
        n_active = update_dual(n, y, wi, z, lambda, &div_zi[0], &ab[0], &b[0]);

        // Something has gone very wrong (probably Nan input)
        if (n_active < 0) {
            // Return Failure Code
            *iter = local_iter;
            return -1;
        }

        update_primal(n, x, y, wi, z, lambda);
        Dx(n, x, &diff_x[0]);

        /*************************** Update Partition
         * ****************************/

        // Count, evaluate (fitness), and store violators
        n_vio = locate_violators(n, z, lambda, &diff_x[0], &vio_index[0],
            &vio_fitness[0], &vio_sort[0]);

        // Update safeguard queue and proportion of violators to be reassigned
        if (n_vio < min_queue) {
            // inflate proprtion
            p = min(delta_e * p, 1);
            // push new min into queue
            vio_queue[queue_index] = n_vio;
            min_queue = n_vio;
            min_queue_index = queue_index;
            // if max val in queue was replaced, compute new max
            if (queue_index == max_queue_index) {
                max_queue = 0;
                for (int j = 0; j < m; j++) {
                    if (vio_queue[j] > max_queue) {
                        max_queue = vio_queue[j];
                        max_queue_index = j;
                    }
                }
            }
            // increment queue index
            queue_index = (queue_index + 1) % m;
        } else if (n_vio >= max_queue) {
            // deflate proprtion
            p = max(delta_s * p, 1 / n_vio);
        } else {
            // push new value into queue
            vio_queue[queue_index] = n_vio;
            // if max/min val in queue was replaced, compute new max/min
            if (queue_index == max_queue_index) {
                max_queue = 0;
                for (int j = 0; j < m; j++) {
                    if (vio_queue[j] > max_queue) {
                        max_queue = vio_queue[j];
                        max_queue_index = j;
                    }
                }
            } else if (queue_index == min_queue_index) {
                min_queue = n;
                for (int j = 0; j < m; j++) {
                    if (vio_queue[j] < min_queue) {
                        min_queue = vio_queue[j];
                        min_queue_index = j;
                    }
                }
            }
            // increment queue index
            queue_index = (queue_index + 1) % m;
        }

        // Print Status
        if (verbose) {
            fprintf(stderr, "|%4d|%9d|%6d|%4.2f|\n", *iter, n_vio, n_active, p);
        }

        // Check Termination Criterion -- Algorithm Converged
        if (n_vio == 0) {
            if (verbose) {
                fprintf(stderr, "Solved\n");
            }
            *iter = local_iter;
            return 1; // Return Success Code
        }

        // Sort violator indices by fitness value
        std::sort(&vio_sort[0], &vio_sort[n_vio], FitnessComparator(&vio_fitness[0]));

        // Reassign first p * n_vio violators
        n_vio = max(static_cast<int>(round(p * n_vio)), 1);
        reassign_violators(n_vio, z, &vio_index[0], &vio_sort[0]);
        *iter = local_iter;
    }

    // Algorithm Failed To Converge
    if (verbose) {
        fprintf(stderr, "MAXITER Exceeded.\n");
    }

    return 0; // Return Failure To Converge Code
}

/******************************************************************************
 *                                Subproblems                                 *
 ******************************************************************************/

/* Given the dual variable, updates the primal according to
        x = W^{-1} (y - lamba*D'z)
   Computation is performed in place and within a single O(n)
   loop that computes div_z_i while updating x_i
*/
inline void update_primal(const int n, double* x, const double* y, const double* wi,
    const double* z, const double lambda)
{
    int i;
    // x[0] = (y[0] + z[0] * lambda) / w[0]
    *x++ = *y + (*z * lambda * *wi);
    y++, wi++;
    // x[1] = (y[1] - (z[1] - 2*z[0])* lambda) / w[1]
    *x++ = *y + ((*(z + 1) - *z - *z) * lambda * *wi);
    y++;
    wi++;
    for (i = 2; i < n - 2; i++, y++, wi++, z++) { // i = 2,...,n-3
        // x[i] = (y[i] + (z[i-2] + z[i] - 2*z[i-1]) * lambda) / w[i]
        *x++ = *y + ((*z - *(z + 1) - *(z + 1) + *(z + 2)) * lambda * *wi);
    }
    // x[n-2] = (y[n-2] + (z[n-4] - 2*z[n-3]) * lambda) / w[n-2]
    *x++ = *y + ((*z - *(z + 1) - *(z + 1)) * lambda * *wi);
    y++;
    wi++;
    z++;
    // x[n-1] = (y[n-1] + z[n-3] * lambda) / w[n-1]
    *x = *y + (*z * lambda * *wi);
}

/* Given a partition dual, update the active set according to
   D[A] * inv(W) * D[A]' z[A] = D[A](y - lambda * inv(W) * D[I]' * z[I]) /
   lambda Quindiagonal matrix solved with LAPACKE's dpbsv interface
   */
int update_dual(const int n, const double* y, const double* wi, double* z,
    const double lambda, double* div_zi, double* ab, double* b)
{
    /* Initiliaze Counters */
    int i, ik;
    int previous = -3;
    int two_previous = -3;
    int k = n - 2; // start with all dual coordinates active

    /* Compute div_zi = inv(W)*D[I]'*z[I] and count active coordinates */
    div_zi[0] = 0;
    div_zi[1] = 0;

    for (i = 0; i < n - 2; i++) {
        div_zi[i + 2] = 0;
        if (z[i] == 1 || z[i] == -1) {
            k--; // Remove inactive coordinate from active count
            div_zi[i] -= z[i];
            div_zi[i + 1] += 2 * z[i];
            div_zi[i + 2] -= z[i];
        }
        div_zi[i] *= wi[i];
    }

    div_zi[n - 2] *= wi[n - 2];
    div_zi[n - 1] *= wi[n - 1];

    /* compute diagonals of D[A] inv(W) D[A]^T and targets = D[A] resid */
    ik = 0;
    for (i = 0; i < n - 2; i++) {
        if (z[i] != 1 && z[i] != -1) {
            /* Update Diag Matrix Content */

            // All main diag inner products eval to 1/wi + 4/wi+1 + 1/wi+2
            ab[ik + 2 * k] = wi[i] + 4 * wi[i + 1] + wi[i + 2];

            // off diag: If previous element in A is off by 1 or 2, inner
            // product evals to -4 or 1 resp
            if (i - previous == 1) {
                ab[ik + k] = -2 * wi[i] - 2 * wi[i + 1];
                // 2nd off diag: If element in A is off by two, inner product
                // evals to 1
                if (i - two_previous == 2) {
                    ab[ik] = wi[i];
                } else {
                    ab[ik] = 0;
                }
            } else if (i - previous == 2) {
                ab[ik + k] = wi[i];
                ab[ik] = 0;
            } else {
                ab[ik + k] = 0;
                ab[ik] = 0;
            }

            // Update Counters For Previous Time Element Was In A
            two_previous = previous;
            previous = i;

            /* Update target content */
            b[ik] = ((y[i + 1] + y[i + 1] - y[i] - y[i + 2]) / lambda) - div_zi[i + 1] - div_zi[i + 1] + div_zi[i] + div_zi[i + 2];

            /* Update active set counter */
            ik++;
        }
    }

    // compute matrix solve
    int result = LAPACKE_dpbsv(LAPACK_ROW_MAJOR, 'U', k, 2, 1, ab, k, b, 1);
    if (result != 0) {
        fprintf(stderr, "LAPACKE: %d\n", result);
        if (result < 0) {
            return -1; // Invalid input: abort
        }
    }

    // update zA
    ik = 0;
    for (i = 0; i < n - 2; i++) {
        if (fabs(z[i]) != 1) {
            z[i] = b[ik++];
        }
    }

    return k;
}

/* locate, count and eval fitness of violators */
int locate_violators(const int n, const double* z, const double lambda,
    const double* diff_x, int* vio_index, double* vio_fitness,
    int* vio_sort)
{
    int n_vio = 0; // number of violations located

    for (int i = 0; i < n - 2; i++) {
        switch (int(z[i])) {
        case 1:
            if (diff_x[i] < 0) {
                vio_index[n_vio] = i;
                vio_fitness[n_vio] = max(lambda * fabs(diff_x[i]), 1);
                vio_sort[n_vio] = n_vio;
                n_vio++;
            }
            break;
        case -1:
            if (diff_x[i] > 0) {
                vio_index[n_vio] = i;
                vio_fitness[n_vio] = max(lambda * fabs(diff_x[i]), 1);
                vio_sort[n_vio] = n_vio;
                n_vio++;
            }
            break;
        default:
            if (fabs(z[i]) > 1) {
                vio_index[n_vio] = i;
                vio_fitness[n_vio] = max(lambda * fabs(diff_x[i]), fabs(z[i]));
                vio_sort[n_vio] = n_vio;
                n_vio++;
            }
            break;
        }
    }
    return n_vio;
}

/* locate, count and eval fitness of violators */
void reassign_violators(const int n_vio, double* z, const int* vio_index,
    const int* vio_sort)
{
    for (int i = 0; i < n_vio; i++) {
        if (fabs(z[vio_index[vio_sort[i]]]) == 1) {
            z[vio_index[vio_sort[i]]] = 0;
        } else {
            if (z[vio_index[vio_sort[i]]] > 1) {
                z[vio_index[vio_sort[i]]] = 1;
            } else if (z[vio_index[vio_sort[i]]] < -1) {
                z[vio_index[vio_sort[i]]] = -1;
            }
        }
    }
}
