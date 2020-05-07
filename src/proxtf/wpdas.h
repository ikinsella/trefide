#ifndef WPDAS_H
#define WPDAS_H

#define likely(x)       __builtin_expect((x),1)
#define unlikely(x)     __builtin_expect((x),0)

/******************************************************************************
 *                         Custom Data Structures                             *
 ******************************************************************************/

/* Comparator passed to c++ sort function in order to sort violator indices
 * in descending order of violator fitness.
 */
struct FitnessComparator {
    const double *fitness_arr;

    FitnessComparator(const double *vio_fitness) : fitness_arr(vio_fitness) {}

    inline bool operator()(int i1, int i2) {
        return fitness_arr[i1] > fitness_arr[i2];
    }
};

/*******************************************************************************
 *                            Function Declaration                             *
 *******************************************************************************/

/* x = y - lambda*D'*z */
void update_primal(int n, double *x, const double *y, const double *wi,
                   const double *z, double lambda);

/* z_a : D_a*D_a'*z_a == D_a * ( (y / lambda) - D_i' z_i)
 * potential optimization opportunity:
 * precomp D_a*(y/lambda) and use trick for D_a*D_i'*z_i
 */
int update_dual(const int n, const double *y, const double *wi, double *z,
                const double lambda, double *div_zi, double *ab, double *b);

inline int locate_violators(const int n, const double *z, const double lambda,
                     const double *diff_x, int *vio_index, double *vio_fitness,
                     int *vio_sort);

inline void reassign_violators(const int n_vio, double *z, const int *vio_index,
                        const int *vio_sort);

/* main routine for pdas l1tf solver */
short weighted_pdas(const int n, const double *y, const double *wi,
                    const double lambda, double *x, double *z, int *iter,
                    double p, const int m, const double delta_s,
                    const double delta_e, const int maxiter, const int verbose);

#endif /* WPDAS_H */
