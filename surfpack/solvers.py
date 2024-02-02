"""
This is where the Picard solver is implemented. See Functional.equilibrium_density_profile, and sandbox/saft_equlibrium.py
for example usage.

Todo: Consider implementing more efficient solvers. The current solver is slow as shit.
"""
import copy
import time

from scipy.constants import Boltzmann, gas_constant
import numpy as np
import warnings
from surfpack.profile import Profile
from surfpack.grid import Grid
from collections import deque
from collections.abc import Iterable
from scipy.constants import Avogadro

class EquilibriumResult:

    def __init__(self, profile, converged, res, i, solver, bad_convergence, max_iter, tol):
        self.profile = profile
        self.z = None
        self.converged = converged
        self.residual = res
        self.tol = tol
        self.iterations = i
        self.max_iter = max_iter
        self.solver = solver
        self.bad_convergence = bad_convergence
        if converged is True:
            self.message = 'Finished with convergence.'
        elif self.iterations == self.max_iter:
            self.message = f'Exited after reaching max number of iterations ({self.max_iter}).'
        elif self.bad_convergence is True:
            self.message = f'Exited due to bad convergence'
        else:
            self.message = f"Exited, but I don't know why..."

    def __repr__(self):
        grid = self.profile[0].grid
        r = 'EquilibriumResult\n'
        r += f'Solver     : {self.solver}\n'
        r += f'profile    : {len(self.profile)} Profiles with\n' \
             f'             Grid : N_points : {grid.N}, Geometry : {grid.geometry}, domain size : {grid.L} Å\n'
        r += f'converged  : {self.converged}\n'
        r += f'residual   : {self.residual} / Tolerance : {self.tol}\n'
        r += f'iterations : {self.iterations} / Max iterations : {self.max_iter}\n'
        r += f'message    : {self.message}'
        return r

    def __str__(self):
        return self.__repr__()

def eq_condition(functional, rho_b, T, p0, Vext=None):
    r"""
    Compute the array $\tilde{\rho}_i(z) = \rho_i^b \exp [- \beta V_i^{ext}(z) + c_i^{(1)}(z) + \beta \mu_i]$

    Args:
        functional (Functional) : The Functional
        p0 (list[Profile]) : The input density profiles
        rho_b (Iterable) : Bulk densities
        T (float) : Temperature
        Vext (callable, optional) : External potential

    Returns:
        list[Profile] : rho_tilde
    """
    Vext = functional.sanitize_Vext(Vext)
    mu = functional.residual_chemical_potential(rho_b, T)
    c = functional.correlation(p0, T)
    beta = 1 / (Boltzmann * T)

    rho_tilde = Profile.zeros_like(p0)
    for i in range(len(rho_tilde)):
        rho_tilde[i] = rho_b[i] * np.exp(- beta * Vext[i](p0[0].grid.z) + c[i] + beta * mu[i])

    return rho_tilde

def fixpoint_rhoT(func, p0, rho_b, T, Vext=None):
    r"""
    Compute the array $\tilde{\rho}_i(z) = \rho_i^b \exp [- \beta V_i^{ext}(z) + c_i^{(1)}(z) + \beta \mu_i]$,
    That is: The fixpoint equilibrium condition at given bulk density and temperature.

    Args:
        func (Functional) : The Functional
        p0 (list[Profile]) : The input density profiles
        rho_b (Iterable) : Bulk densities
        T (float) : Temperature
        Vext (callable, optional) : External potential

    Returns:
        list[Profile] : rho_tilde
    """
    Vext = func.sanitize_Vext(Vext)
    mu = func.residual_chemical_potential(rho_b, T)
    c = func.correlation(p0, T)
    beta = 1 / (Boltzmann * T)

    rho_fix = Profile.zeros_like(p0)
    for i in range(len(rho_fix)):
        rho_fix[i] = rho_b[i] * np.exp(- beta * Vext[i](p0[0].grid.z) + c[i] + beta * mu[i])

    return rho_fix

def root_rhoT(func, p0, rho_b, T, Vext=None):
    """
    Evaluate the equlibrium condition f(x) = 0
    That is: The root-finder equilibrium condition at given bulk density and temperature.

    Args:
        func (Functional) : The Functional
        p0 (list[Profile]) : The input density profiles
        rho_b (Iterable) : Bulk densities
        T (float) : Temperature
        Vext (callable, optional) : External potential

    Returns:
        list[Profile] : rho_tilde
    """
    rho_fix = fixpoint_rhoT(func, p0, rho_b, T, Vext=Vext)
    rho_root = Profile.zeros_like(p0)
    for i in range(len(rho_root)):
        rho_root[i] = rho_fix[i] - p0[i]
    return rho_root

def fixpoint_NT(func, p0, z0, N, T, Vext):
    c = func.correlation(p0, T)
    beta = 1 / (Boltzmann * T)
    p_fix = Profile.zeros_like(p0)
    z_fix = np.zeros(func.ncomps)
    grid = p0[0].grid

    for i in range(len(p0)):
        exp_factor = np.exp(- beta * Vext[i](p0[0].grid.z) + c[i])
        exp_factor = Profile(exp_factor, grid)
        p_fix[i] = z0[i] * exp_factor
        z_fix[i] = N[i] / exp_factor.integrate()

    return p_fix, z_fix

def root_NT(func, p0, z0, N, T, Vext):
    p_fix, z_fix = fixpoint_NT(func, p0, z0, N, T, Vext=Vext)
    z_root = z_fix - z0
    p_root = Profile.zeros_like(p_fix)
    for i in range(len(p_root)):
        p_root[i] = p_fix[i] - p0[i]
    return p_root, z_root

def fixpoint_muT(func, p0, exp_dmu, N, T, Vext):
    c = func.correlation(p0, T)
    beta = 1 / (Boltzmann * T)
    p_fix = Profile.zeros_like(p0)
    grid = p0[0].grid
    debroglie = np.array([1 / (func.eos.de_broglie_wavelength(i + 1, T) * 1e10) ** 3 for i in range(func.ncomps)])

    exp_factors = Profile.zeros_like(p0)
    E = np.zeros(func.ncomps)

    z_denominator = 0
    for i in range(func.ncomps):
        exp_factors[i] = Profile(np.exp(c[i] - beta * Vext[i](p0[0].grid.z)), grid)
        E[i] = exp_factors[i].integrate()
        z_denominator += E[i] * debroglie[i] * exp_dmu[i]

    for i in range(len(p0)):
        z = N * debroglie[i] * exp_dmu[i] / z_denominator
        p_fix[i] = z * exp_factors[i]

    return p_fix

def root_muT(func, p0, z0, exp_dmu, N, T, Vext):
    p_fix, exp_dmu_fix = fixpoint_muT(func, p0, z0, exp_dmu, N, T, Vext)
    p_root = Profile.zeros_like(p_fix)
    exp_dmu_root = exp_dmu_fix - exp_dmu
    for i in range(len(p_root)):
        p_root[i] = p_fix[i] - p0[i]
    return p_root, exp_dmu_root

def picard_rhoT(func, p0, rho_b, T, Vext=None, max_iter=100, tol=1e-8, mixing_alpha=0.1, verbose=False):
    """
    Do a series of Picard steps, if a step fails, reduce the mixing parameter and retry

    Args:
        func (Functional) : The functional
        p0 (list[Profile]) : Initial profiles
        rho_b (1d array) : Bulk densities
        T (float) : Temperature [K]
        Vext (callable) : External potential
        max_iter (int) : Maximum number of iterations
        tol (float) : tolerance for convergence
        mixing_alpha (float) : Picard iteration mixing parameter
        verbose (bool) : Whether to print info

    Returns:
        EquilibriumResult : The profiles after iterations
    """
    Vext = func.sanitize_Vext(Vext)
    grid = p0[0].grid

    def fixpoint(x_k):
        p_k = [Profile(x_k[grid.N * i: grid.N * (i + 1)], grid) for i in range(func.ncomps)]
        p_fix = fixpoint_rhoT(func, p_k, rho_b, T, Vext=Vext)
        x_fix = np.concatenate(p_fix)
        return x_fix

    x0 = np.concatenate(p0)
    sol = picard(fixpoint, x0, max_iter=max_iter, tol=tol, mixing_alpha=mixing_alpha, verbose=verbose)
    sol.profile = [Profile(sol.profile[grid.N * i: grid.N * (i + 1)], grid) for i in range(func.ncomps)]
    return sol

def picard_NT(func, p0, N, T, Vext=None, max_iter=100, tol=1e-8, mixing_alpha=0.1, verbose=False):
    Vext = func.sanitize_Vext(Vext)
    grid = p0[0].grid

    def fixpoint(x_k):
        p_k = [Profile(x_k[grid.N * i : grid.N * (i + 1)], grid) for i in range(func.ncomps)]
        z_k = x_k[grid.N * func.ncomps:]
        p_fix, z_fix = fixpoint_NT(func, p_k, z_k, N, T, Vext=Vext)
        x_fix = np.concatenate((np.concatenate(p_fix), z_fix))
        return x_fix

    z0 = func.fugacity(p0, T) / (Boltzmann * T)
    x0 = np.concatenate((np.concatenate(p0), z0))
    sol = picard(fixpoint, x0, max_iter=max_iter, tol=tol, mixing_alpha=mixing_alpha, verbose=verbose)
    sol.profile = [Profile(sol.profile[grid.N * i : grid.N * (i + 1)], grid) for i in range(func.ncomps)]
    return sol

def picard_muT(func, p0, mu, N, T, Vext=None, max_iter=100, tol=1e-8, mixing_alpha=0.1, verbose=False):
    Vext = func.sanitize_Vext(Vext)
    grid = p0[0].grid

    exp_dmu = np.exp((mu - mu[0]) / (Boltzmann * T))

    def fixpoint(x_k):
        p_k = [Profile(x_k[grid.N * i : grid.N * (i + 1)], grid) for i in range(func.ncomps)]
        p_fix = fixpoint_muT(func, p_k, exp_dmu, N, T, Vext=Vext)
        x_fix = np.concatenate(p_fix)
        return x_fix

    x0 = np.concatenate(p0)
    sol = picard(fixpoint, x0, max_iter=max_iter, tol=tol, mixing_alpha=mixing_alpha, verbose=verbose)
    sol.profile = [Profile(sol.profile[grid.N * i : grid.N * (i + 1)], grid) for i in range(func.ncomps)]

    return sol

def anderson_rhoT(func, p0, rho_b, T, Vext=None, prev_res=None, prev_x=None, tol=1e-10, m_max=50, max_iter=200,
                  beta_mix=0.05, ensure_positive=True, verbose=False):

    Vext = func.sanitize_Vext(Vext)
    grid = p0[0].grid
    ncomps = len(rho_b)
    get_profiles = lambda x_k: [Profile(x_k[i * grid.N : (i + 1) * grid.N], grid) for i in range(ncomps)]
    def residual(x_k, T, Vext):
        eq = eq_condition(func, rho_b, T, [Profile(x_k[i * grid.N : (i + 1) * grid.N], grid) for i in range(ncomps)], Vext)
        return np.array([eq[i] - x_k[i * grid.N : (i + 1) * grid.N] for i in range(ncomps)]).flatten()

    x = np.asarray(p0).flatten()
    sol = anderson(residual, x, fargs=(T, Vext), prev_res=prev_res, prev_x=prev_x, tol=tol, m_max=m_max,
                   max_iter=max_iter, beta_mix=beta_mix, ensure_positive=ensure_positive, verbose=verbose)

    sol.profile = get_profiles(sol.profile)
    return sol

def anderson_NT(func, p0, N, T, Vext=None, prev_res=None, prev_x=None, tol=1e-10, m_max=50, max_iter=200,
                  beta_mix=0.05, ensure_positive=True, verbose=False):
    Vext = func.sanitize_Vext(Vext)
    z0 = func.fugacity(p0, T) / (Boltzmann * T)
    grid = p0[0].grid
    def residual(x_k):
        profiles_ = [Profile(x_k[i * grid.N : (i + 1) * grid.N], grid) for i in range(func.ncomps)]
        z_ = x_k[func.ncomps * grid.N:]
        p_root, z_root = root_NT(func, profiles_, z_, N, T, Vext)
        x_root = np.concatenate((np.asarray(p_root).flatten(), z_root))
        return x_root

    x = np.concatenate((np.asarray(p0).flatten(), z0))
    sol = anderson(residual, x, prev_res=prev_res, prev_x=prev_x, tol=tol, m_max=m_max,
                   max_iter=max_iter, beta_mix=beta_mix, ensure_positive=ensure_positive, verbose=verbose)
    profiles = [Profile(sol.profile[i * grid.N : (i + 1) * grid.N], grid) for i in range(func.ncomps)]
    z = sol.profile[func.ncomps * grid.N:]
    sol.profile = profiles
    sol.z = z
    return sol

def anderson_muT(func, p0, mu, N, T, Vext=None, prev_res=None, prev_x=None, tol=1e-10, m_max=50, max_iter=200,
                  beta_mix=0.05, ensure_positive=True, verbose=False):
    Vext = func.sanitize_Vext(Vext)
    grid = p0[0].grid
    exp_dmu = np.exp((mu - mu[0]) / (Boltzmann * T))

    def residual(x_k):
        p_k = [Profile(x_k[grid.N * i: grid.N * (i + 1)], grid) for i in range(func.ncomps)]
        p_fix = fixpoint_muT(func, p_k, exp_dmu, N, T, Vext=Vext)
        x_fix = np.concatenate(p_fix)
        return x_fix - x_k

    x0 = np.concatenate(p0)
    sol = anderson(residual, x0, prev_res=prev_res, prev_x=prev_x, tol=tol, m_max=m_max,
                   max_iter=max_iter, beta_mix=beta_mix, ensure_positive=ensure_positive, verbose=verbose)
    profiles = [Profile(sol.profile[i * grid.N : (i + 1) * grid.N], grid) for i in range(func.ncomps)]
    z = sol.profile[func.ncomps * grid.N:]
    sol.profile = profiles
    sol.z = z
    return sol

def picard(fixpoint, x0, fargs=(), fkwargs=None, max_iter=100, tol=1e-8, mixing_alpha=0.1, verbose=False):
    if fkwargs is None:
        fkwargs = {}
    converged = False
    bad_convergence = False
    success = True
    i = 0
    res = 0
    # If we hit a divergent state, we will fall back to this value, and reduce mixing.
    # The fallback value is updated every fallback_update_freq iterations.
    fallback_update_freq = 5000
    x_fallback = copy.deepcopy(x0)
    res_fallback = 0
    while (converged is False) and (i < max_iter):
        x_next = fixpoint(x0, *fargs, **fkwargs)
        if any(np.isnan(x_next)):
            success = False
            x0 = copy.deepcopy(x_fallback)

        res = np.linalg.norm(x0 - x_next) / np.sqrt(len(x0))
        if success is False:
            if mixing_alpha > 1e-3:
                if verbose > 0:
                    print(f'Picard Mixing appears to be too agressive after {i} iterations. Reducing to : {mixing_alpha}')
                    print(f'and falling back to solution at {i - (i % fallback_update_freq)} iterations, where residual is {res_fallback}.')

                mixing_alpha /= 2  # Dampen iteration and try again
                i -= i % fallback_update_freq
                success = True
                continue
            else:
                warnings.warn('Could not converge Picard!', RuntimeWarning, stacklevel=2)
                bad_convergence = True
                break
        elif res < tol:
            converged = True

        if i % fallback_update_freq == 0:
            x_fallback = copy.deepcopy(x0)
            res_fallback = res

        x0 = x0 * (1 - mixing_alpha) + x_next * mixing_alpha
        i += 1


    return EquilibriumResult(x0, converged, res, i, 'Picard', bad_convergence, max_iter, tol)

def anderson(residual, p0, fargs=(), fkwargs=None, prev_res=None, prev_x=None, tol=1e-10, m_max=50, max_iter=200,
             beta_mix=0.05, ensure_positive=True, verbose=False):
    if fkwargs is None:
        fkwargs = {}

    if prev_res is None:
        prev_res = deque([])
        prev_x = deque([])
    else:
        prev_res = deque(prev_res)
        prev_x = deque(prev_x)

    converged = False
    bad_convergence = False
    x = p0
    for k in range(max_iter):

        if len(prev_res) > m_max:
            prev_res.popleft()
            prev_x.popleft()

        res = residual(x, *fargs, **fkwargs)

        if any(np.isnan(res.flatten())):
            bad_convergence = True

        if bad_convergence is True:
            if verbose > 0:
                print('Anderson mixing failed using parameters:')
                print(f'tol : {tol}, beta_mix : {beta_mix}, m_max : {m_max}')
                print(f'Residual at previous iteration ({k}) : {np.linalg.norm(prev_res[-1]) / np.sqrt(len(prev_res[-1]))}')
                print(f'If this is being run as part of a sequential solver, the sequential solver may handle the issue '
                      f'by falling back to another solver routine.\nIn that case, running the sequential solver with '
                      f'verbose > 0 can yield more information about whats going on.')
            x = prev_x[-1]
            break

        if np.linalg.norm(res.flatten()) / np.sqrt(len(res.flatten())) < tol:
            converged = True
            break

        prev_res.append(res)
        prev_x.append(np.copy(x))

        m = len(prev_res)

        # calculate alpha
        r = np.ones((m+1, m+1))
        r[m, m] = 0.0
        r[:-1, :-1] = np.dot(prev_res, np.transpose(prev_res))
        alpha = np.zeros(m + 1)
        alpha[m] = 1.0
        alpha = np.linalg.solve(r, alpha)

        x *= 0
        for i in range(m):
            x += alpha[i] * (prev_x[i] - beta_mix * prev_res[i])

        if ensure_positive:
            x = abs(x)

    # profile, converged, res, i, solver, bad_convergence, max_iter, tol
    if converged is True:
        sol = EquilibriumResult(x, converged, np.linalg.norm(res) / np.sqrt(len(res)), k + 1,
                                'Anderson', bad_convergence, max_iter, tol)
    else:
        sol = EquilibriumResult(x, converged, np.linalg.norm(res) / np.sqrt(len(res)), k + 1,
                                'Anderson', bad_convergence, max_iter, tol)
    return sol

def ng_extrapolation(functional, rho_b, T, p0, Vext=lambda z : 0, mixing_alpha=0.05, max_iter=5, tol=1e-10):

    fn_m2 = p0
    gn_m2 = eq_condition(functional, rho_b, T, fn_m2, Vext=Vext)
    fn_m1 = [(1 - mixing_alpha) * fi + mixing_alpha * gi for fi, gi in zip(fn_m2, gn_m2)]
    gn_m1 = eq_condition(functional, rho_b, T, fn_m1, Vext=Vext)
    fn = [(1 - mixing_alpha) * fi + mixing_alpha * gi for fi, gi in zip(fn_m1, gn_m1)]
    gn = eq_condition(functional, rho_b, T, fn, Vext=Vext)

    f = deque([fn_m2, fn_m1, fn])
    g = deque([gn_m2, gn_m1, gn])
    d = deque([[gi - fi for fi, gi in zip(fn, gn)] for fn, gn in zip(f, g)])

    fn_p1 = Profile.zeros_like(fn)
    grid = fn[0].grid
    dz = grid.dz
    converged = False
    ncomps = len(p0)
    for k in range(max_iter):
        for ci in range(ncomps):
            dn = d[-1][ci]
            dn_m1 = d[-2][ci]
            dn_m2 = d[-3][ci]

            d01 = dn - dn_m1
            d02 = dn - dn_m2

            d12 = np.trapz(d01 * d02, dx=dz)

            D = [[np.trapz(d01 * d01, dx=dz), d12],
                 [d12, np.trapz(d02 * d02, dx=dz)]]

            b = [np.trapz(dn * d01, dx=dz), np.trapz(dn * d02, dx=dz)]
            c = np.linalg.solve(D, b)

            fn_p1[ci] = Profile((1 - c[0] - c[1]) * gn[ci] + c[0] * gn_m1[ci] + c[1] * gn_m2[ci], grid=grid)

        gn_p1 = [Profile(gi, grid) for gi in eq_condition(functional, rho_b, T, fn_p1, Vext=Vext)]
        dn_p1 = [gi - fi for fi, gi in zip(fn_p1, gn_p1)]

        res = 0
        for di in dn_p1:
            res += np.linalg.norm(di) / np.sqrt(len(di))

        if np.isnan(res):
            warnings.warn(f'NG extrapolation failed after {k + 1} iterations.', RuntimeWarning)
            fn_p1 = f[0]
            dn_p1 = d[0]
            res = 0
            for di in dn_p1:
                res += np.linalg.norm(di) / np.sqrt(len(di))
            k -= 2
            break

        if res < tol:
            converged = True
            break

        f.popleft()
        f.append(copy.deepcopy(fn_p1))
        g.popleft()
        g.append(copy.deepcopy(gn_p1))
        d.popleft()
        d.append(copy.deepcopy(dn_p1))


    sol = EquilibriumResult(fn_p1, converged, res, k + 1, 'NG extrapolation')
    return sol

def solve_sequential_NT(func, p0, N, T, solvers, tolerances, Vext=None, solver_kwargs=None, verbose=False):
    """
    Forwards call to `sequential_solver`. Only functions as a nice interface when (N, T) are the constraints

    Args:
        func (Functional) : The Functional
        p0 (list[Profile]) : Initial guess for density profiles [particles / Å]
        N (list) : Total particle number of each species
        T (float) : Temperature [K]
        solvers (list[callable]) : List of individual solver routines to call
        tolerances (list[float]) : List of tolerance for each individual solver
        Vext (list[callable]) : External potential experienced by each component
        solver_kwargs (list[dict]) : Kwargs to pass to each solver
        verbose (bool) : If true, output information about progress during run

    Returns:
        EquilibriumResult : The Equilibrium density profiles, along with information about convergence, iteration number etc.
    """
    Vext = func.sanitize_Vext(Vext)
    return solve_sequential(func, p0, (N, T, Vext), solvers, tolerances, solver_kwargs=solver_kwargs, verbose=verbose)

def solve_sequential_muT(func, p0, mu, N, T, solvers, tolerances, Vext=None, solver_kwargs=None, verbose=False):
    """
    Forwards call to `sequential_solver`. Only functions as a nice interface when (N, T) are the constraints

    Args:
        func (Functional) : The Functional
        p0 (list[Profile]) : Initial guess for density profiles [particles / Å]
        N (float) : Total number of particles
        T (float) : Temperature [K]
        mu (1d array) : Chemical potential of each species [J / particle]
        solvers (list[callable]) : List of individual solver routines to call
        tolerances (list[float]) : List of tolerance for each individual solver
        Vext (list[callable]) : External potential experienced by each component
        solver_kwargs (list[dict]) : Kwargs to pass to each solver
        verbose (bool) : If true, output information about progress during run

    Returns:
        EquilibriumResult : The Equilibrium density profiles, along with information about convergence, iteration number etc.
    """
    # Vext = func.sanitize_Vext(Vext)
    return solve_sequential(func, p0, (mu, N, T), solvers, tolerances, solver_kwargs=solver_kwargs, verbose=verbose)

def solve_sequential_rhoT(func, p0, rho_b, T, solvers, tolerances, solver_kwargs=None, Vext=None, verbose=False):
    Vext = func.sanitize_Vext(Vext)
    return solve_sequential(func, p0, (rho_b, T, Vext), solvers, tolerances, solver_kwargs=solver_kwargs, verbose=verbose)

def solve_sequential(func, p0, constraints, solvers, tolerances, solver_kwargs=None, verbose=0, max_fallbacks=5, default_action=None):
    """
    Sequential solver for equilibrium density profile
    Works by iterating through the list `solvers`: Each solver is run until converging to the corresponding tolerance
    in the list `tolerances`. If a solver diverges, an attempt to fall back to a previous solver is attempted. If
    a solver does not converge within the maximum number of iterations, the user is prompted to handle the case
    manually.
    Typical use case is:
    solvers = [picard_NT, picard_NT, anderson_NT]
    tolerances = [1e-3, 1e-5, 1e-9]
    solver_kwargs = [{'alpha_mix' : 0.01, 'max_iter' : 200},
                     {'alpha_mix' : 0.05, 'max_iter' : 200},
                     {'mix_beta' : 0.02, 'max_iter': 100}]
    Such that the solvers can become more aggressive as the solution is approached.

    Args:
        func (Functional) : The Functional
        p0 (list[Profile]) : Initial guess for density profiles [particles / Å]
        constraints (tuple) : Constrained variables (e.g. (N, T), (mu, T) or (rho_b, T))
        solvers (list[callable]) : List of individual solver routines to call
        tolerances (list[float]) : List of tolerance for each individual solver
        solver_kwargs (list[dict]) : Kwargs to pass to each solver
        verbose (int) : Output information about progress during run, larger number gives more output
        max_fallbacks (int) : Maximum number of times to fallback the solver before prompting the user for guidance.

    Returns:
        EquilibriumResult : The Equilibrium density profiles, along with information about convergence, iteration number etc.
    """
    if solver_kwargs is None:
        solver_kwargs = [{} for _ in solvers]

    solver_idx, tol_idx = 0, 0
    res = tolerances[0]
    n_iter = 0
    n_fallbacks = 0
    while res > tolerances[-1]:
        tol = tolerances[solver_idx]

        solver = solvers[solver_idx]
        sol = solver(func, p0, *constraints, tol=tol, verbose=verbose - 1, **solver_kwargs[solver_idx])
        n_iter += sol.iterations

        if sol.residual > res: # Detecting divergent behaviour (residual has increased since previous solver)
            sol.bad_convergence = True

        if (sol.converged is False) and (sol.bad_convergence is True):
            """
            If bad_convergence is True the solver has diverged. In that case: Fall back to the previous solver, and
            run that solver with the geometric mean tolerance of this solver and the previous solver, then retry this solver.
            """
            if verbose > 0:
                print('#' * 30)
                print(f'Sequential solver did not converge when using: {sol.solver}, (nr. {solver_idx})')
                print(f'With {solver_kwargs[solver_idx]}')
                print(f'Tolerance : {tol}')
                print(f'After {sol.iterations} iterations, residual is {sol.residual}\n')
            if solver_idx > 0:
                if n_fallbacks >= max_fallbacks:
                    print(f'Reached Maximum number of fallbacks ({n_fallbacks}), choose an action:')
                    if default_action is None:
                        choice = input('(e)xit the solver / try (m)more fallbacks / (t)hrow ValueError')
                    else:
                        choice = default_action
                        print(f'Using default action : {default_action}')

                    if choice == 'e':
                        break
                    elif choice == 'm':
                        max_fallbacks += int(input('Enter new number of fallbacks:'))
                    elif choice == 't':
                        raise ValueError('Triggered by selected choice in non-convergent solver.')

                tolerances[solver_idx - 1] = np.sqrt(tol * tolerances[solver_idx - 1])
                solver_idx -= 1
                n_fallbacks += 1
                if verbose > 0:
                    print(f'Falling back to : {solvers[solver_idx]}')
                    print(f'With {solver_kwargs[solver_idx]}')
                    print(f'Tolerance : {tolerances[solver_idx]}')
                    print('#'*30)
                continue
            warnings.warn('Sequential solver could not converge! Call the solver with verbose=True for debugging info.',
                          RuntimeWarning, stacklevel=2)
            if default_action == 't':
                raise ValueError('Triggered by non-convergent solver')
            break
        elif (sol.converged is False) and (sol.bad_convergence is False): # Manually handling cases that behave badly
            print('#' * 30)
            print(f'Sequential solver did not converge when using: {sol.solver}, (nr. {solver_idx})')
            print(f'With {solver_kwargs[solver_idx]}')
            print(f'Tolerance : {tol}')
            print(f'After {sol.iterations} iterations, residual is {sol.residual}\n')
            if default_action is None:
                choice = input('Choose an action: r/s/f/e/t \n'
                               '(r)etry the current solver / (s)skip the current solver / (f)all back to previous solver / (e)xit solver / (t)hrow ValueError)\n')
            else:
                choice = default_action
                print(f'Using default action : {default_action}')

            if choice == 'r':
                p0 = sol.profile
                continue
            elif choice == 's':
                pass
            elif choice == 'e':
                break
            elif choice == 'f':
                solver_idx -= 1
                continue
            elif choice == 't':
                raise ValueError("Triggered by selected action in non-convergent solver.")
        elif (sol.converged is True) and (verbose > 0):
                print('#' * 50)
                print(f'Solver {solver} (nr. {solver_idx})')
                print(f'With {solver_kwargs[solver_idx]}')
                print(f'Converged after {sol.iterations} (tol : {tol}, residual : {sol.residual})')
                print('#' * 50)

        p0 = sol.profile
        res = sol.residual
        solver_idx += 1
        if tol_idx < solver_idx:
            tol_idx = solver_idx

    sol.iterations = n_iter
    return sol


class SequentialSolver:

    def __init__(self, spec, constraints=None, solvers=None, tolerances=None, solver_kwargs=None):
        self.solvers = [] if (solvers is None) else solvers
        self.tolerances = [] if (tolerances is None) else tolerances
        self.solver_kwargs = [] if (solver_kwargs is None) else solver_kwargs
        self.constraints = constraints
        self.spec = spec
        self.max_fallbacks = 5
        self.default_action = None

        if spec == 'muT':
            self.picard = picard_muT
            self.anderson = anderson_muT
        elif spec == 'NT':
            self.picard = picard_NT
            self.anderson = anderson_NT
        elif spec == 'rhoT':
            self.picard = picard_rhoT
            self.anderson = anderson_rhoT
        else:
            raise KeyError(f"Invalid specification key : {spec}. Valid specs are 'muT', 'NT' and 'rhoT'.")

    def set_max_fallbacks(self, max_fallbacks):
        self.max_fallbacks = max_fallbacks

    def set_default_action(self, default):
        self.default_action = default

    def set_constraints(self, constraints):
        self.constraints = constraints

    def add_picard(self, tol, mixing_alpha=0.05, **kwargs):
        self.solvers.append(self.picard)
        self.add_tol_kwargs(tol, mixing_alpha=mixing_alpha, **kwargs)

    def add_anderson(self, tol, beta_mix=0.05, **kwargs):
        self.solvers.append(self.anderson)
        self.add_tol_kwargs(tol, beta_mix=beta_mix, **kwargs)

    def add_tol_kwargs(self, tol, **kwargs):
        if len(self.tolerances) > 0:
            if tol >= self.tolerances[-1]:
                warnings.warn(f'Adding solver with tol : {tol}. Previous tolerance is {self.tolerances[-1]}', Warning,
                              stacklevel=2)
        self.tolerances.append(tol)
        self.solver_kwargs.append(kwargs)

    def truncate_idx(self, idx):
        solvers = self.solvers[:idx]
        tolerances = self.tolerances[:idx]
        solver_kwargs = self.solver_kwargs[:idx]
        constraints = self.constraints
        return SequentialSolver(self.spec, constraints, solvers=solvers, tolerances=tolerances, solver_kwargs=solver_kwargs)

    def truncate_tol(self, tol):
        for i in range(len(self.tolerances)):
            if self.tolerances[i] < tol:
                return self.truncate_idx(i - 1)
        return self.truncate_idx(-1)

    def __call__(self, func, p0, verbose=0):
        return solve_sequential(func, p0, self.constraints, self.solvers, self.tolerances, solver_kwargs=self.solver_kwargs, verbose=verbose,
                                max_fallbacks=self.max_fallbacks, default_action=self.default_action)

class GridRefiner:

    def __init__(self, solvers=None, grids=None):
        self.solvers = solvers if (solvers is not None) else []
        self.grids = grids if (grids is not None) else []
        self.nsteps = len(grids) if (grids is not None) else 0

    @staticmethod
    def init_nsteps(basesolver, start_grid, end_grid, n_steps, tol_limits=None):
        n_gridpoints = np.linspace(start_grid.N, end_grid.N + 1, n_steps, dtype=int)
        grids = [Grid(n_grid, end_grid.geometry, end_grid.L) for n_grid in n_gridpoints]

        tol_limits = -1 if (tol_limits is None) else tol_limits
        if not isinstance(tol_limits, Iterable):
            tol_limits = [tol_limits for _ in range(n_steps - 1)]
            tol_limits.append(-1)

        solvers = [basesolver.truncate_tol(tol) for tol in tol_limits]
        return GridRefiner(solvers, grids)

    @staticmethod
    def init_twostep(basesolver, coarsegrid, finegrid, tol=None):
        return GridRefiner.init_nsteps(basesolver, coarsegrid, finegrid, 2, tol_limits=tol)

    def __call__(self, func, p0, verbose=False):
        p0 = Profile.lst_on_grid(p0, self.grids[0])
        p0 = EquilibriumResult(p0, False, np.nan, 0, 'None', False, 0, -1) # Packing the initial guess into an EquilibriumResult, because the inside of the following
                                                                           # loop treats p0 as an EquilibriumResult, not a list[Profile]
        for grid, solver in zip(self.grids, self.solvers):
            if verbose is True: print(f'Running grid with {grid.N} points...')
            p0 = Profile.lst_on_grid(p0.profile, grid)
            p0 = solver(func, p0, verbose=verbose)
        return p0
