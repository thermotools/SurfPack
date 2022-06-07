#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from collections import deque

def anderson_acceleration(residual, x0, mmax=50, beta=0.05,
                          tolerance=1.0e-10, max_iter=200,
                          log_iter=False, n_positive=None):
    """ Method solving Picard iteration with Anderson acceleration
    """
    converged = False
    resm = deque([])
    xm = deque([])
    x_sol = np.zeros_like(x0)
    x_sol[:] = x0[:]
    for k in range(max_iter):
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
        r[m,m] = 0.0
        for i in range(m):
            for j in range(m):
                r[i,j] = np.dot(resm[i], resm[j])
        alpha = np.zeros(m + 1)
        alpha[m] = 1.0
        # Solve using LU from lapack
        alpha = np.linalg.solve(r, alpha)
        #print("alpha:",alpha[0:m], sum(alpha[0:m]))

        # update solution
        x_sol[:] = 0.0
        for i in range(m):
            x_sol[:] += alpha[i] * (xm[i][:] - beta * resm[i][:])

        if n_positive is not None:
            # Only positive densities possible
            # Not the case for chemical potentials
            x_sol[:n_positive] = np.abs(x_sol[:n_positive])

        # check for convergence
        resv = resm[m - 1]
        res = np.linalg.norm(resv) / np.sqrt(len(resv))

        if log_iter:
            print("Anderson mixing {:>4} | {:.6e}".format(k, res))

        if np.isnan(res):
            print("Anderson Mixing failed")

        if res < tolerance:
            converged = True
            break
    return x_sol, converged

if __name__ == "__main__":
    print("dft_numerics")
