import numpy as np
from solvers.lagrangian import pdas, pdas_ws


def interpolate(y,
                delta,
                lam=None,
                z_hat=None,
                k=1,
                step_heuristic=60,
                maxiter=2000,
                verbose=0,
                tol=1e-3):
    """ Solves constrained problem for detrended signals"""

    # Compute first search point and step size
    # computation of scale assumes detrended & centered
    scale = delta / np.sqrt(np.mean(np.power(y, 2) - delta))
    int_min = np.log(3*scale + 1)
    int_width = np.log(20+(1/scale)) - np.log(3+(1/scale))
    tau_int = int_width / step_heuristic
    if lam is None:
        # Init in middle of transformed interval
        lam = np.exp(int_width / 2 + int_min) - 1

    # Compute initial solution
    if z_hat is None:
        x_hat, z_hat, iters = pdas(y, lam, maxiter, verbose)
    else:
        x_hat, z_hat, iters = pdas_ws(y, lam, z_hat, maxiter, verbose)

    # Evaluate solution
    mse = np.mean(np.power(y - x_hat, 2))
    err = np.abs(delta - mse) / delta
    direction = np.sign(delta - mse)

    # After k-interps, transition to small stepping
    for _ in range(k):

        # Terminate if landed in tol band
        if err <= tol:
            return x_hat, z_hat, lam

        # Take step to compute slope in transformed space
        lam = np.exp(np.log(lam + 1) + direction * tau_int) - 1
        x_hat, z_hat, iter_ = pdas_ws(y, lam, z_hat, maxiter, verbose)
        mse_prev = mse
        mse = np.mean(np.power(y - x_hat, 2))
        err = np.abs(delta - mse) / delta
        iters += iter_

        # Interpolate in transformed space
        slope = direction * (mse - mse_prev) / tau_int
        tau_interp = (delta - mse) / slope
        lam = np.exp(np.log(lam + 1) + tau_interp) - 1

        # Solve for interpolated lambda
        x_hat, z_hat, iter_ = pdas_ws(y, lam, z_hat, maxiter, verbose)

        # Evaluate current position
        mse = np.mean(np.power(y - x_hat, 2))
        err = np.abs(delta - mse) / delta
        iters += iter_
        direction = np.sign(delta - mse)

    # Continue stepping until cross over or into tol band
    while direction * np.sign(delta - mse) > 0 and err > tol:

        # Fixed step size in transformed space
        lam = np.exp(np.log(lam + 1) + direction * tau_int) - 1

        # Resolve every step
        x_hat, z_hat, iter_ = pdas_ws(y, lam, z_hat, maxiter, verbose)

        # Evaluate current position
        mse_prev = mse
        mse = np.mean(np.power(y - x_hat, 2))
        err = np.abs(delta - mse) / delta
        iters += iter_

    # Interpolate for final fit
    slope = direction * (mse - mse_prev) / tau_int
    tau_interp = (delta - mse) / slope
    lam = np.exp(np.log(lam + 1) + tau_interp) - 1

    # Fit with final lambda
    x_hat, z_hat, iter_ = pdas.warm_start(y, lam, z_hat, maxiter, verbose)

    # Evaluate final search point
    mse = np.mean(np.power(y - x_hat, 2))
    err = np.abs(delta - mse)/delta
    iters += iter_

    # return final computed solution
    return x_hat, z_hat, lam, iters
