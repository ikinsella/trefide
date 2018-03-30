import time
import sys
import numpy as np
import scipy
import scipy.sparse
import scipy.linalg
import scipy.sparse.linalg


""" wrapper for primal dual solvers """


def lagrangian_l1tf(signal,
                    lam,
                    warm_start=None,
                    solver="first order",
                    # Shared
                    buff=5,
                    maxiter=2e4,
                    verbose=False,
                    record=False,                    
                    # First Order
                    criterion='gap',
                    tol=1e-4,
                    theta=0,
                    # Active Set
                    p=1,
                    delta_s=.9,
                    delta_e=1.1,
                    safeguard=True):
    """
    Solves Lagrangian Form of L1TF Denoising Problem With A Primal Dual Solver:
    argmin_x (lambda/2)||signal - x||_2^2 + ||Dx||_1
    """
    if solver == 'first order':
        T = len(signal)

        # Assign Regularization Functions
        reg_eval = lambda z: l1_norm(z) / T
        reg_conj = lambda z: linf_indicator(z, r=1)
        reg_prox = lambda z, sigma: linf_projection(z, r=1)

        # Assign Loss Functions
        loss_eval = lambda x: squared_error(x, signal) / (lam * T)
        loss_conj = lambda x: squared_error_conj(signal, x, lam) / T
        loss_prox = lambda x, tau: squared_error_prox(x, signal, tau, lam)

        # Assign Warm Start
        if warm_start is None:
            warm_start = (signal, init_dual(signal, signal, lam))
        elif len(warm_start) is not 2:
            warm_start = (warm_start, init_dual(warm_start, signal, lam))

        # Minimize Objective
        x, z, results, converged = first_order(init=warm_start,
                                               L=diff_norm(len(signal)),
                                               K=diff,
                                               Kc=div,
                                               F=reg_eval,
                                               Fc=reg_conj,
                                               prox_Fc=reg_prox,
                                               G=loss_eval,
                                               Gc=loss_conj,
                                               prox_G=loss_prox,
                                               theta=theta,
                                               tau=.01,
                                               tol=tol,
                                               buff=buff,
                                               maxiter=maxiter,
                                               miniter=10,
                                               accel=False,
                                               criterion=criterion,
                                               verbose=verbose,
                                               record=record)

    elif solver == 'active set':

        # Assign Warm Start
        if warm_start is None:
            warm_start = init_partition(signal)
        elif len(warm_start) is not 3:
            warm_start = init_partition(warm_start)

        # Minimize Objective
        x, z, results, converged = active_set(init=warm_start,
                                              y=signal,
                                              lam=lam,
                                              p=p,
                                              m=buff,
                                              delta_s=delta_s,
                                              delta_e=delta_e,
                                              maxiter=maxiter,
                                              safeguard=safeguard,
                                              verbose=verbose,
                                              record=record)
        
    return x, z, results, converged


""" Active Set Solver """


def active_set(init,
               y,
               lam,
               p=1,
               m=5,                           
               delta_s=.9,
               delta_e=1.1,
               maxiter=1e5,
               safeguard=True,
               verbose=False,
               record=False):
    """
    """

    # Initialize partitions, dual, and queue
    P, N, A = init
    z = np.zeros(len(y)-2)
    if safeguard:
        Q = [0]*m
    if record:
        prog = np.zeros([int(maxiter), 4])

    # Main Iterations
    t0 = time.time()
    q_itr = 0
    for itr in np.arange(1, maxiter+1).astype('int'):

        # Update Dual & Minimize Subspace
        z[P] = 1
        z[N] = -1
        theta, z = subspace_minimization(y, lam, z, A, np.sort(np.append(P, N)))
        
        # Locate Violations
        VP = np.argwhere(diff_subset(theta, P) < 0)
        nVP = len(VP)
        VN = np.argwhere(diff_subset(theta, N) > 0)
        nVN = len(VN)        
        VAP = np.argwhere(z[A] > 1)
        nVAP = len(VAP)
        VAN = np.argwhere(z[A] < -1)
        nVAN = len(VAN)
        V = np.append(np.append(P[VP], N[VN]),
                      np.append(A[VAP], A[VAN]))
        nV = len(V)

        if safeguard:  # Check Safeguard Queue For Cycles
            if (nV >= max(Q)) and (itr > m):
                p = max(delta_s * p, 1 / nV)
            elif (nV < min(Q)) and (itr > m):
                Q[q_itr % m] = nV
                q_itr += 1
                p = min(delta_e * p, 1)
            else:
                Q[q_itr % m] = nV
                q_itr += 1

        # Report Progress
        if verbose:
            sys.stdout.write('||Iter:%5d|'%itr)
            sys.stdout.write('| |V|:%5d|'%nV)
            sys.stdout.write('| |A|:%5d|'%len(A))
            sys.stdout.write('|p:%10.3e|'%p)            
            #sys.stdout.write('|Obj:%10.3e|'%obj[itr % buff])
            #sys.stdout.write('|Dual:%10.3e|'%dual_obj)
            #sys.stdout.write('|Gap:%10.3e|'%delta_gap)            
            sys.stdout.write('|CPU:%10.3e||\n'%(time.time() - t0))
            sys.stdout.flush()

        if record:
            prog[itr-1,:] = np.array([nV, len(A), p, time.time() - t0])

        # check termination criterion
        if nV == 0:
            # Proper Termination
            if record:
                prog = prog[:itr,:]
            else:
                prog = np.array([nV, len(A), p, time.time() - t0])

            return theta, z, ((P, N, A), prog), True

        # Move top k violators
        k = int(round(p * nV))

        fitness = np.maximum(lam * np.abs(diff_subset(theta, V)),
                             np.abs(z[V]))
        move_idx = np.argsort(fitness * -1)[:k]  # only move largest k indices

        # Update violator partitions to only keep those to be moved
        VP = VP[move_idx[move_idx < nVP]]
        VN = VN[move_idx[np.logical_and(move_idx >= nVP,
                                           move_idx < nVP + nVN)]
                   - nVP]
        VAP = VAP[move_idx[np.logical_and(move_idx >= nVP + nVN,
                                             move_idx < nV - nVAN)]
                     - (nVP + nVN)]
        VAN = VAN[move_idx[move_idx >= nV - nVAN] - (nV - nVAN)]

        # Update partitions with chosen violators
        PVP = P[VP]
        P = np.sort(np.append(np.delete(P, VP), A[VAP]))
        NVN = N[VN]
        N = np.sort(np.append(np.delete(N, VN), A[VAN]))
        A = np.sort(np.append(np.delete(A, np.append(VAP, VAN)),
                              np.append(PVP, NVN)))

    # Maxiter Reached
    if not record:
        prog = np.array([nV, len(A), p, time.time() - t0])
        
    return theta, z, ((P, N, A), prog), False


def subspace_minimization(y, lam, z, A, I):
    """ subspce minimization step from ___ """
    if len(A) == 0:
        #print("lA = 0")
        pass
    elif len(A) == 1:
        #print("lA = 1")
        z[A] = diff_subset(y - lam * div_subset(z[I], I, len(y)), A) / (6* lam)
    else:
        # z[A] = scipy.linalg.solve_banded(
            # (2,2),
            # diff_subset_gram(A).data,
            # diff_subset(y - lam * div_subset(z[I], I, len(y)), A) / lam
        # )
        z[A] = scipy.linalg.solveh_banded(
            diff_subset_gram(A).data,
            diff_subset(y - lam * div_subset(z[I], I, len(y)), A) / lam,
            lower=False
        )        
        
    theta = y - lam * div(z)
    return theta, z


""" first order solver """


def first_order(init, L, K, Kc,
                F, Fc, prox_Fc,
                G, Gc, prox_G,
                theta=1,
                tau=.01,
                gamma=None,
                tol=1e-4,
                buff=5,
                maxiter=1e4,
                miniter=10,
                # primal=True ,
                accel=False,
                criterion='obj',
                verbose=False,
                record=False):
    """ Chambolle's Primal-Dual Algorithm For Solving: argmin_x F(Kx) + G(x)
    % For Graph TV-Regularization these functions will be:
    % K: Difference Operator
    % Kc: Divergence Operator
    % F: L1L2 norm
    % Fc: Unit LinfL2 Ball Indicator Function
    % prox_Fc: Projection onto the unit LinfL2 ball
    % G: Loss Function of (x, y, A, lambda)
    % Gc: Conjugate Loss Function of (x, y, A, lambda)
    % prox_G: Prox operator for loss function
    % gamma = .7 * lambda;
    """

    """ Input Parsing """

    # Functionals & Optimization Variables
    x, y = init
    x_bar = x

    # Opt Routine Tuning Params
    if accel:  # Tau & Sigma Adaptive
        if gamma is None:
            raise Exception('If "accel" is chosen, gamma must be provided')
        tau = 1 / L;
        sigma = 1 / L;
    else:  # Tau & Sigma Fixed
        if tau is None:
            raise exception('If "accel" is not chosen, tau must be provided.')
        sigma = 1 / (tau * (L ** 2))

    # Stopping Criterion Params & Buffers
    obj = np.zeros(buff);
    delta_x = np.zeros(buff);
    delta_y = np.zeros(buff);
    converged = False
    eval_criterion = {
        'obj': lambda delta : delta[0] < tol,
        'x': lambda delta: np.max(delta[1]) < tol,
        'y': lambda delta : np.max(delta[2]) < tol,
        'all': lambda delta: np.max([delta[0],
                                     np.max(delta[1]),
                                     np.max(delta[2])]) < tol,
        'any': lambda delta: np.min([delta[0],
                                     np.max(delta[1]),
                                     np.max(delta[2])]) < tol,
        'gap': lambda delta: delta[-1] < tol
        }[criterion]
    if record:
        prog = np.zeros([int(maxiter), 7])

    """ Main Iterations """
    t0 = time.time()  # start the clock
    for itr in np.arange(1, int(maxiter+1)):  

        # Perform Updates
        x_old = x
        y_old = y
        y = prox_Fc(y + (sigma * K(x_bar)), sigma) # sigma won't affect TF, can be removed
        x = prox_G(x - (tau * Kc(y)), tau)
        if accel:
            theta = 1 / np.sqrt(1 + (2 * gamma * tau))
            tau = theta * tau
            sigma = sigma / theta
        x_bar = x + (theta * (x - x_old))
        obj[itr % buff] = G(x) + F(K(x))
        dual_obj = -1 * (Gc(-Kc(y)) + Fc(y))

        # Evaluate Progress
        delta_obj = np.max(np.abs((obj[itr % buff] - obj) / obj))
        delta_x[itr % buff] = np.linalg.norm((x - x_old) / (x_old + np.finfo(np.double).tiny))
        delta_y[itr % buff] = np.linalg.norm((y - y_old) / (y_old + np.finfo(np.double).tiny))
        delta_gap = obj[itr % buff] - dual_obj
        t = time.time() - t0
        # Report Progress
        if verbose:
            sys.stdout.write('||Iter:%d|'%itr);
            sys.stdout.write('|Obj:%10.3e|'%obj[itr % buff])
            sys.stdout.write('|Dual:%10.3e|'%dual_obj)
            sys.stdout.write('|Gap:%10.3e|'%delta_gap)            
            sys.stdout.write('|Delta Obj:%10.3e|'%delta_obj)
            sys.stdout.write('|Delta x:%10.3e|'%np.max(delta_x))
            sys.stdout.write('|Delta y:%10.3e|'%np.max(delta_y))
            sys.stdout.write('|CPU:%10.3e||\n'%t)
            sys.stdout.flush()

        if record:
            prog[itr-1:,:] = np.array([obj[itr % buff],
                                       dual_obj,
                                       delta_gap,
                                       delta_obj,
                                       delta_x[itr % buff],
                                       delta_y[itr % buff],
                                       t])

        # Check Stopping Criterion
        converged = eval_criterion((delta_obj, delta_x, delta_y, delta_gap))

        if converged and (itr > miniter):
            break

    """ End Main Iterations """

    """ Summarize & Report Results """
    time_elapsed = time.time() - t0
    obj = G(x) + F(K(x))
    dual_obj = -1 * (Gc(-Kc(y)) + Fc(y))
    loss = G(x)
    reg = F(K(x))

    if verbose:  
        sys.stdout.write('\nFinished the main algorithm!  Results:\n')
        sys.stdout.write('Number of iterations = %d\n'%itr)
        sys.stdout.write('Primal Objective: G(x) + F(K(x, E)) = %10.3e\n'%obj)
        sys.stdout.write('Dual Objective: -(Gc(-Kc(y, E)) + Fc(y)) = %10.3e\n'%dual_obj)        
        sys.stdout.write('Loss: G(x) = %10.3e\n'%loss)
        sys.stdout.write('Regularization: F(K(x, E)) = %10.3e\n'%reg)
        sys.stdout.write('CPU time = %10.3e\n'%time_elapsed)
        sys.stdout.flush()

    if record:
        prog = prog[:itr,:]
    else:
        prog = np.array([obj[itr % buff],
                         dual_obj,
                         delta_gap,
                         delta_obj,
                         delta_x[itr % buff],
                         delta_y[itr % buff],
                         time_elapsed])
    return x, y, (prog,), converged


""" Utilities """


# LossEval = @(x) SquaredError(x, y) / Lambda;
def squared_error(y, x):
    """
    # %% Squared Error Loss Function
    # % Usage: [ loss ] = SquaredError(y, x)
    # % loss = (1/2) * || y - x ||_2^2
    # % Input:
    # % x: Current Estimate : [P, 1]
    # % y: Observations : [N, 1]
    # % Returns:
    # % loss : Value Of Loss Function At Current Estimate
    """
    return .5 * np.sum(np.power(y - x, 2));

                
# ConjEval = @(x) SquaredErrorConj(x, y, Lambda);
def squared_error_conj(y, x, lam):
    """
    Fenchel Conjugate Of The Sqaured Error Loss Function
    Usage: [loss] = SquaredErrorConj(y, x, lambda)
    F*(z) = sup_x <z,x> - (1 /(2 * lambda))||y - x||_2^2
    F*(x) = <x,y> + (lambda / 2)||x||_2^2
    Input:
        x: Current Estimate : [P, 1]
        y: Observations : [N, 1]
        lambda : Resularization Constant : [scalar]
    Returns:
        loss : Value Of Conjugate Loss Function At Current Estimate
    """
    return x.T.dot(y) + ((lam / 2) * np.sum(np.power(x,2)))


# ProxEval = @(x, tau) SquaredErrorProx(x, y, tau, Lambda);
def squared_error_prox(x, y, tau, lam):
    """
    Squared Error Prox Operator
    Usage : [x_tilde] = SquaredErrorProx(x, y, tau, lambda)
    x~ = prox_G(x) = argmin_x~ lambda * ||x - x~||_2^2 + tau * ||y - x~||_2^2
    G(x) = (tau/(2*lambda))||y - x||_2^2
    Input:
       x: current estimate : [P, 1]
       lambda: regularization param : [scalar]
       tau: opt routine param : [scalar]
    Returns:
       x~: denoised estimate: [P, 1]
    """
    return ((lam * x) + (tau * y)) / (tau + lam)  # weighted average
                

# RegEval = @L1;
def l1_norm(y):
    """L_1 Norm Evaluation
    ||diff(x)_e||_1
    Usage: GFL(x) = F(K(x)) = L1(diffE(x))
    Input:
	Y: Diff Matrix - Rows diff wrt each edge for single vertex : [E,P]
    Returns:
        loss: Sum of variation at each edge of graph
    """
    return  np.sum(np.abs(y));


# ConjEval = @(Y) Ic_LInf(Y, 1);
def linf_indicator(y, r=1):
    """
    Indicator of the L_Inf Ball
    i_c[||.||_inf](b)
    Usage: GFLc(Y) = Fc(Y) = Ic_LInf(Y, 1)
    Input:
	Y: Dual Variable : [E]
	r: Radius of L_inf ball
    Returns:
        loss: 0 if Y is within L_Inf ball of radius b, realmax otherwise
    """
    # unit L_inf indicator for each variable, but since it follows the
    # projection it should always return 0 in practice...    
    return (np.max(np.abs(y)) > (r+1e-3)) * np.finfo(np.double).max


# ProxEval = @(Y, sigma) Proj_LInf(Y, 1); % Prox_Fc
def linf_projection(y, r=1):
    """ 
    L_Inf Norm Projection Operator:
    F(Y) <- ||Y||_1
    Y~i = prox_F*(Y)_i = Y_i / max(1, |Y_i|) 
    Usage: Prox_GFLc(x) = Prox_Fc(Y) = Proj_[L_Inf(1)](Y)
    Input:
    Y: Current estimate for dual variable : [P,P]
    Returns:
    Y_tilde: Projection of each variable onto the unit L_Inf Ball
    """
    return y / np.maximum(1, np.abs(y) / r);


def diff(x):
    """ O(n) eval of second order difference op"""
    return 2 * x[1:-1] - x[:-2] - x[2:]


def div(y):
    """ O(n) eval of second order divergence op"""
    return np.append(np.append([-1 * y[0], 2 * y[0] - y[1]],
                               diff(y)),
                     [2 * y[-1] - y[-2], -1 * y[-1]])

def diff_norm(n):
    """ norm of second order diff op for vectors length n"""
    #D = scipy.sparse.diags([-1 * np.ones(n-1),
    #                        2 * np.ones(n),
    #                        -1 * np.ones(n-1)], [-1, 0 1]).tocsr()[1:-1,:]
    #return scipy.sparse.linalg.svds(D, k=1)[1][0]  # max singular value
    return 4  # TODO: too slow to be worth being precise, using upper bound


def diff_subset(theta, A):
    """ 
    Evaluate subset of rows (A) of the discrete 
    second order difference matrix against a vector theta....
    Assume that A is sorted
    """
    return (2 * theta[A+1]) - theta[A] - theta[A+2]


def div_subset(zI, I, n):
    """   """
    theta = np.zeros(n)
    theta[I] -= zI
    theta[I+1] += 2 * zI
    theta[I+2] -= zI
    return theta


def diff_subset_gram(A):
    """
    Computed D_A D_A^T for a set of indices A 
    Assume A sorted increasing
    """
    n = len(A)
    a = np.ones(n) * 6.0
    b = np.zeros(n-1)
    adj = A[1:] - A[:-1]
    b[adj == 1] = -4
    b[adj == 2] = 1
    c = np.zeros(n-2)
    c[A[2:] - A[:-2] == 2] = 1
    #return scipy.sparse.diags([c, b, a, b, c], [-2, -1, 0, 1, 2]).transpose()
    return scipy.sparse.diags([c, b, a], [-2, -1, 0]).transpose()


def init_partition(x, tol=1e-3):
    """ Assign an initial partition from a given primal variable x """
    P = np.argwhere(diff(x) > tol)
    N = np.argwhere(diff(x) < -tol)
    A = np.argwhere(np.logical_and(diff(x) <= tol, 
                                   diff(x) >= -tol))
    return (P.flatten(), N.flatten(), A.flatten())


def init_dual(x, y, lam):
    """ Assign an initial dual variable from a given primal variable x """
    P, N, A = init_partition(x)
    z = np.zeros(len(x) - 2)
    z[P] = 1
    z[N] = -1
    _, z = subspace_minimization(y, lam, z, A, np.sort(np.append(P, N))) 
    return linf_projection(z)
