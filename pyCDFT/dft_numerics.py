#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from collections import deque


def anderson_acceleration(residual, x0, mmax=50, beta=0.05,
                          tolerance=1.0e-10, max_iter=200,
                          log_iter=False, ensure_positive_x=True):
    """ Method solving Picard iteration with Anderson acceleration
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


def picard_iterations(residual, x0, max_rel_change=1.0,
                      tolerance=1.0e-10, max_iter=200, beta=0.15,
                      log_iter=False, ensure_positive_x=False):
    """ Method solving Picard iteration
    """
    if log_iter:
        print(f"solver           iter | residual | beta")

    converged = False
    x_sol = np.zeros_like(x0)
    x_sol[:] = x0[:]
    res = np.zeros_like(x_sol)
    beta_vec = np.zeros_like(x_sol)
    for k in range(1, max_iter+1):
        # Calculate residual
        res[:] = residual(x_sol)
        # calculate beta
        for i in range(len(x0)):
            # Avoid too big relative change
            beta_i = max_rel_change * \
                np.abs(x_sol[i]) / np.max(np.abs(res[i]), 1.0e-10)
            beta_vec[i] = np.min(beta, beta_i)

        # update solution
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
