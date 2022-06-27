#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from collections import deque


PICARD = 1
ANDERSON = 2


class dft_solver_param():
    """
    """

    def __init__(self, algorithm=ANDERSON, beta=0.05, tolerance=1.0e-10, max_iter=200,
                 ensure_positive_x=True, ng_frequency=None, mmax=50):
        """Paramater class for dft solver

        Args:
            algorithm (int, optional): Algorithm to use (PICARD or ANDERSON). Defaults to ANDERSON.
            beta (float, optional): Fraction of damping factor for successive iteration. Defaults to 0.05.
            tolerance (float, optional): Residual tolerance. Defaults to 1.0e-10.
            max_iter (int, optional): Maximum number of iterations. Defaults to 200.
            ensure_positive_x (bool, optional): Reset negative x values?. Defaults to True.
            ng_frequency (int, optional): When to do Ng extrapolations. Used with Picard solver. Typical value 10. Defaults to None.
            mmax (int, optional): How many iterations to include in the Anderson scheme. Defaults to 50.
        """
        self.algorithm = algorithm
        self.max_rel_change = 1.0
        self.beta = beta
        self.tolerance = tolerance
        self.max_iter = max_iter
        self.ensure_positive_x = ensure_positive_x
        self.ng_frequency = ng_frequency
        self.mmax = mmax


class dft_solver():
    """Solver class for classical DFT
    """

    def __init__(self):
        """
        """
        self.dft_solver_params = []

    def solve(self, x0, residual, log_iter=False):
        """Solve residual(x) = 0.0, given x0 as initial guess

        Args:
            x0 (np.ndarray): Variable vector
            residual (function): Function taking x as input and returining residual
            log_iter (bool, optional): Print iterations. Defaults to False.
        """

        # Add default solver if none specified
        if not self.dft_solver_param:
            self.dft_solver_param.append(dft_solver_param())

        x_sol = np.zeros_like(x0)
        x_sol[:] = x0[:]
        for sp in self.dft_solver_param:
            if sp.solver == PICARD:
                x_sol[:], converged = picard_iteration(residual, x_sol, max_rel_change=sp.max_rel_change,
                                                       tolerance=sp.tolerance, max_iter=sp.max_iter, beta=sp.beta,
                                                       log_iter=log_iter, ensure_positive_x=sp.ensure_positive_x,
                                                       ng_frequency=sp.ng_frequency)
            elif sp.solver == ANDERSON:
                x_sol[:], converged = anderson_acceleration(residual, x_sol, mmax=sp.mmax, beta=sp.beta,
                                                            tolerance=sp.tolerance, max_iter=sp.max_iter,
                                                            log_iter=log_iter, ensure_positive_x=sp.ensure_positive_x)

        return x_sol, converged

    def picard(self,
               tolerance=1.0e-10,
               max_iter=200,
               beta=0.15,
               ensure_positive_x=False,
               ng_frequency=None):
        """ Set up solver paramaters for Picard iterations

        Args:
            tolerance (float, optional): Residual tolerance. Defaults to 1.0e-10.
            max_iter (int, optional): Maximum number of iterations. Defaults to 200.
            beta (float, optional): Fraction of damping factor for successive iteration. Defaults to 0.15.
            ensure_positive_x (bool, optional): Reset negative x values?. Defaults to True.
            ng_frequency (int, optional): When to do Ng extrapolations. Used with Picard solver. Typical value 10. Defaults to None.
        """
        self.dft_solver_param.append(dft_solver_param(solver=PICARD,
                                                      beta=beta,
                                                      tolerance=tolerance,
                                                      max_iter=max_iter,
                                                      ensure_positive_x=ensure_positive_x,
                                                      ng_frequency=ng_frequency))

    def anderson(self,
                 tolerance=1.0e-10,
                 max_iter=200,
                 beta=0.05,
                 ensure_positive_x=False,
                 mmax=50):
        """Set up solver paramaters for Anderson mixing

        Args:
            tolerance (float, optional): Residual tolerance. Defaults to 1.0e-10.
            max_iter (int, optional): Maximum number of iterations. Defaults to 200.
            beta (float, optional): Fraction of damping factor for successive iteration. Defaults to 0.05.
            ensure_positive_x (bool, optional): Reset negative x values?. Defaults to True.
            mmax (int, optional): How many iterations to include in the Anderson scheme. Defaults to 50.
        """
        self.dft_solver_param.append(dft_solver_param(solver=ANDERSON,
                                                      beta=beta,
                                                      tolerance=tolerance,
                                                      max_iter=max_iter,
                                                      ensure_positive_x=ensure_positive_x,
                                                      mmax=mmax))


def anderson_acceleration(residual, x0, mmax=50, beta=0.05,
                          tolerance=1.0e-10, max_iter=200,
                          log_iter=False, ensure_positive_x=True):
    """Method solving Picard iteration with Anderson acceleration

    Args:
        x0 (np.ndarray): Variable vector
        residual (function): Function taking x as input and returining residual
        tolerance (float, optional): Residual tolerance. Defaults to 1.0e-10.
        max_iter (int, optional): Maximum number of iterations. Defaults to 200.
        log_iter (bool, optional): Print iterations. Defaults to False.
        beta (float, optional): Fraction of damping factor for successive iteration. Defaults to 0.05.
        ensure_positive_x (bool, optional): Reset negative x values?. Defaults to True.
        mmax (int, optional): How many iterations to include in the Anderson scheme. Defaults to 50.

    Returns:
        x_sol (np.ndarray): Current solution
        converged (bool): Did the solver converge?
    """
    if log_iter:
        print(f"Anderson mixing: iter | res | alpha")

    converged = False
    resm = deque([])
    xm = deque([])
    x_sol = np.zeros_like(x0)
    x_sol[:] = x0[:]
    for k in range(1, max_iter+1):
        # drop old values
        if len(resm) == mmax:
            resm.popleft()
            xm.popleft()

        m = len(resm) + 1

        # calculate residual
        res = np.zeros_like(x_sol)
        x = np.zeros_like(x_sol)
        x[:] = x_sol[:]
        res[:] = residual(x)
        resm.append(res)
        xm.append(x)

        # calculate alpha
        r = np.ones((m+1, m+1))
        r[m, m] = 0.0
        for i in range(m):
            for j in range(m):
                r[i, j] = np.dot(resm[i], resm[j])
        alpha = np.zeros(m + 1)
        alpha[m] = 1.0
        # Solve using LU from lapack
        alpha = np.linalg.solve(r, alpha)
        # print("alpha:",alpha[0:m], sum(alpha[0:m]))

        # update solution
        x_sol[:] = 0.0
        for i in range(m):
            x_sol[:] += alpha[i] * (xm[i][:] - beta * resm[i][:])

        if ensure_positive_x:
            x_sol[:] = np.abs(x_sol[:])

        # check for convergence
        resv = resm[m - 1]
        res = np.linalg.norm(resv) / np.sqrt(len(resv))

        if log_iter:
            print("Anderson mixing {:>4} | {:.6e} | {:.6e}".format(
                k, res, alpha[m-1]))

        if np.isnan(res):
            print("Anderson Mixing failed")

        if res < tolerance:
            converged = True
            break
    return x_sol, converged


class ng_extrapolation():
    """
    Accelerate Picard solution using extrapolations as described in appendix of Ng(1974):
        Kin-Chue Ng
        Hypernetted chain solutions for the classical one-component plasma up to Γ = 7000
        The Journal of Chemical Physics
        1974. 61(7): 2680-2689
        doi: 10.1063/1.1682399
    """

    def __init__(self, N, n_update):
        """

        Args:
            N(int): Size of arrays
            n_update(int): Update only every n_update iteration
        """
        self.N = N
        if n_update is not None:
            self.n_update = max(n_update, 3)
        else:
            self.n_update = None
        self.gn = np.zeros(N)
        self.gnm1 = np.zeros(N)
        self.gnm2 = np.zeros(N)
        self.dn = np.zeros(N)
        self.dnm1 = np.zeros(N)
        self.dnm2 = np.zeros(N)

    def push_back(self, fn, gn, iteration):
        """

        Args:
            fn(np.ndarray): Variable of iterative method
            gn(np.ndarray): gn = A fn
            iteration(int): Current iteration
        """
        if self.n_update is None:
            return
        # No need to copy arrays that will not be used:
        if iteration % self.n_update == 0 or iteration % self.n_update > self.n_update - 3:
            self.gnm2[:] = self.gnm1[:]
            self.gnm1[:] = self.gn[:]
            self.gn[:] = gn[:]

            self.dnm2[:] = self.dnm1[:]
            self.dnm1[:] = self.dn[:]
            self.dn[:] = self.gn[:] - fn[:]

    def extrapolate(self):
        """
        """
        d01 = self.dn[:] - self.dnm1[:]
        d02 = self.dn[:] - self.dnm2[:]
        a = np.zeros((2, 2))
        b = np.zeros(2)
        # Divide by N to avoid too large numbers
        b[0] = np.inner(self.dn[:], d01)
        b[1] = np.inner(self.dn[:], d02)
        a[0, 0] = np.inner(d01, d01)
        a[0, 1] = np.inner(d01, d02)
        a[1, 0] = a[0, 1]
        a[1, 1] = np.inner(d02, d02)
        scaling = max(np.max(np.abs(b)), 1.0e-3)
        a /= scaling
        b /= scaling
        c = np.linalg.solve(a, b)
        fnp1 = (1 - c[0] - c[1]) * self.gn[:] + \
            c[0] * self.gnm1[:] + \
            c[1] * self.gnm2[:]
        return fnp1

    def time_to_update(self, iteration):
        """

        Args:
            iteration(int): Iteration index

        Returns:
            (bool): True if it is time to update
        """

        if self.n_update is None:
            update = False
        elif iteration < 3:
            update = False
        else:
            update = iteration % self.n_update == 0
        return update


def picard_iteration(residual, x0, max_rel_change=1.0,
                     tolerance=1.0e-10, max_iter=200, beta=0.15,
                     log_iter=False, ensure_positive_x=False,
                     ng_frequency=None):
    """Method solving Picard iteration

    Args:
        x0 (np.ndarray): Variable vector
        residual (function): Function taking x as input and returining residual
        tolerance (float, optional): Residual tolerance. Defaults to 1.0e-10.
        max_iter (int, optional): Maximum number of iterations. Defaults to 200.
        beta (float, optional): Fraction of damping factor for successive iteration. Defaults to 0.15.
        log_iter (bool, optional): Print iterations. Defaults to False.
        ensure_positive_x (bool, optional): Reset negative x values?. Defaults to True.
        ng_frequency (int, optional): When to do Ng extrapolations. Used with Picard solver. Typical value 10. Defaults to None.

    Returns:
        x_sol (np.ndarray): Current solution
        converged (bool): Did the solver converge?
    """
    if log_iter:
        print(f"solver           iter | residual | beta")

    if ng_frequency is not None:
        # Extrapolations according to Ng 1974?
        ng = ng_extrapolation(len(x0), ng_frequency)
        x_new = np.zeros_like(x_sol)

    converged = False
    x_sol = np.zeros_like(x0)
    x_sol[:] = x0[:]
    res = np.zeros_like(x_sol)
    beta_vec = np.zeros_like(x_sol)
    res_avg = 1.0
    for k in range(1, max_iter+1):
        # Calculate residual
        res[:] = residual(x_sol)

        update = True
        if ng_frequency is not None:
            # Update Ng history
            x_new[:] = x_sol[:]-res[:]
            ng.push_back(x_sol, x_new, k)
            if res_avg < 1.0e-3 and ng.time_to_update(k):
                x_sol[:] = ng.extrapolate()
                update = False

        # update solution
        if update:
            # calculate beta
            for i in range(len(x0)):
                # Avoid too big relative change
                beta_i = max_rel_change * \
                    np.abs(x_sol[i]) / np.max(np.abs(res[i]), 1.0e-10)
                beta_vec[i] = np.min(beta, beta_i)
            x_sol[:] -= res[:] * beta_vec[:]

        if ensure_positive_x:
            x_sol[:] = np.abs(x_sol[:])

        res_avg = np.linalg.norm(res) / np.sqrt(len(res))
        if log_iter:
            print("Picard iteration {:>4} | {:.6e} | {}".format(
                k, res, np.min(beta_vec)))

        if np.isnan(res):
            print("Picard iteration failed")

        if res_avg < tolerance:
            converged = True
            break
    return x_sol, converged


if __name__ == "__main__":
    print("dft_numerics")
