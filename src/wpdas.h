#ifndef WPDAS_H
#define WPDAS_H

/******************************************************************************
 *                         Custom Data Structures                             *
 ******************************************************************************/

/* Comparator passed to c++ sort function in order to sort violator indices
 * in descending order of violator fitness.
 */
struct FitnessComparator {
    const double* fitness_arr;

    FitnessComparator(const double* vio_fitness)
        : fitness_arr(vio_fitness)
    {
    }

    inline bool operator()(int i1, int i2)
    {
        return fitness_arr[i1] > fitness_arr[i2];
    }
};

/******************************************************************************
 *                            Function Declaration                            *
 ******************************************************************************/

/**
 * main routine for pdas l1tf solver
 *
 * @param n Data length.
 * @param y Observations.
 * @param wi Inverse observation weights.
 * @param lambda Regularization parameter.
 * @param x Primal variable.
 * @param z Dual variable.
 * @param iter Pointer to iter # (so we can return it).
 * @param p Proportion of violators to reassign.
 * @param m Size of violator history queue.
 * @param delta_s Proportion by which `p` is shrunk.
 * @param delta_e Proportion by which `p` is grown.
 * @param maxiter Max number of outer loop iterations.
 * @param verbose Whether to log to stderr.
 * @return -1 on error, 0 if failed to converge, 1 on success.
 */
short weighted_pdas(const int n, const double* y, const double* wi,
    const double lambda, double* x, double* z, int* iter,
    double p, const int m, const double delta_s,
    const double delta_e, const int maxiter, const int verbose);

/* x = y - lambda*D'*z */
void update_primal(int n, double* x, const double* y, const double* wi,
    const double* z, double lambda);

/* z_a : D_a*D_a'*z_a == D_a * ( (y / lambda) - D_i' z_i)
 * potential optimization opportunity:
 * precomp D_a*(y/lambda) and use trick for D_a*D_i'*z_i
 */
int update_dual(const int n, const double* y, const double* wi, double* z,
    const double lambda, double* div_zi, double* ab, double* b);

int locate_violators(const int n, const double* z, const double lambda,
    const double* diff_x, int* vio_index, double* vio_fitness,
    int* vio_sort);

void reassign_violators(const int n_vio, double* z, const int* vio_index,
    const int* vio_sort);

#endif /* WPDAS_H */
