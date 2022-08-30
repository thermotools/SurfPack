#!/usr/bin/env python3
import numpy as np
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utility import weighted_densities_1D, get_thermopack_model, \
    weighted_densities_pc_saft_1D
from pyctp import pcsaft
from constants import NA, RGAS, Geometry
from scipy.fft import dct, idct, dst, idst, fft, ifft
from scipy.special import spherical_jn
from enum import Enum
from sympy import sympify, lambdify

class WeightFunctionType(Enum):
    # Heaviside step function
    THETA = 1
    # Dirac delta function
    DELTA = 2
    # Vector Dirac delta
    DELTAVEC = 3
    # Normalized Heaviside step function
    NORMTHETA = 4

class WeightFunction(object):

    def __init__(self, wf_type, kernel_radius, alias, prefactor,
                 convolve=True, calc_from=None):
        """

        Args:
        rho_b (ndarray): Bulk densities
        R (ndarray): Particle radius for all components
        """
        assert (convolve==True and calc_from is None) or (convolve==False and calc_from is not None)
        self.wf_type = wf_type
        self.kernel_radius = kernel_radius
        self.alias = alias
        self.prefactor_str = prefactor
        self.prefactor = lambdify(("R"), sympify(prefactor), "numpy")
        self.convolve = convolve
        self.calc_from = calc_from
        # For transformations
        self.prefactor_evaluated = None
        self.R = None
        self.fw = None
        self.w_conv_steady = None
        self.k_grid = None
        self.k_cos_R = None
        self.k_sin = None
        self.k_sin_R = None
        self.one_div_r = None
        self.r = None
        self.geometry = None
        self.fn = None

    @staticmethod
    def Copy(Other):
        """Method used to duplicate WeightFunction for different components

        Args:
        Other (WeightFunction): Other instance of WeightFunction
        """
        return WeightFunction(wf_type=Other.wf_type,
                              kernel_radius=Other.kernel_radius,
                              alias=Other.alias,
                              prefactor=Other.prefactor_str,
                              convolve=Other.convolve,
                              calc_from=Other.calc_from)

    def generate_fourier_weights(self, grid, R):
        """
        """
        self.R = R
        self.geometry = grid.geometry
        self.fn = np.zeros(grid.n_grid)
        self.prefactor_evaluated = self.prefactor(R)
        if self.convolve:
            if self.geometry == Geometry.PLANAR:
                self.generate_planar_fourier_weights(grid, R)
            elif self.geometry == Geometry.SPHERICAL:
                self.generate_spherical_fourier_weights(grid, R)
            elif self.geometry == Geometry.POLAR:
                self.generate_polar_fourier_weights(grid, R)

    def generate_planar_fourier_weights(self, grid, R):
        """
        """
        self.fw = np.zeros(grid.n_grid)
        self.k_cos_R = np.zeros(grid.n_grid)
        self.k_sin = np.zeros(grid.n_grid)
        self.k_sin_R = np.zeros(grid.n_grid)
        N = grid.n_grid
        L = grid.domain_size
        self.k_cos_R = 2 * np.pi * np.linspace(0.0, N - 1, N) / (2 * L) * R * self.kernel_radius
        self.k_sin = 2 * np.pi * np.linspace(1.0, N, N) / (2 * L)
        self.k_sin_R = self.k_sin * R * self.kernel_radius

        if self.wf_type == WeightFunctionType.THETA:
            self.w_conv_steady = (4.0/3.0)*np.pi*((R*self.kernel_radius)**3)
            self.fw[:] = 4/3 * np.pi * (R*self.kernel_radius)**3 * \
                (spherical_jn(0, self.k_cos_R) + spherical_jn(2, self.k_cos_R))
        elif self.wf_type == WeightFunctionType.NORMTHETA:
            self.w_conv_steady = 1.0
            self.fw[:] = spherical_jn(0, self.k_cos_R) + spherical_jn(2, self.k_cos_R)
        elif self.wf_type == WeightFunctionType.DELTA:
            # The convolution of the fourier weights for steady profile
            self.w_conv_steady = 4.0*np.pi*(R*self.kernel_radius)**2
            self.fw[:] = 4.0 * np.pi * (R*self.kernel_radius)**2 * spherical_jn(0, self.k_cos_R)
        elif self.wf_type == WeightFunctionType.DELTAVEC:
            # The convolution of the fourier weights for steady profile
            self.w_conv_steady = 0.0
            self.fw[:] = self.k_sin * \
            (4.0/3.0 * np.pi * (R*self.kernel_radius)**3 * (spherical_jn(0, self.k_sin_R) + spherical_jn(2, self.k_sin_R)))

    def generate_spherical_fourier_weights(self, grid, R):
        """
        """
        self.R = R
        R_kern = R * self.kernel_radius
        self.fw = np.zeros(grid.n_grid)
        self.k_cos_R = np.zeros(grid.n_grid)
        self.k_sin = np.zeros(grid.n_grid)
        self.k_sin_R = np.zeros(grid.n_grid)
        N = grid.n_grid
        L = grid.domain_size
        self.k_sin = 2 * np.pi * np.linspace(1.0, N, N) / (2 * L)
        self.k_sin_R = self.k_sin * R_kern
        self.one_div_r = (1/grid.z)
        self.r = grid.z
        if self.wf_type == WeightFunctionType.THETA:
            self.w_conv_steady = (4.0/3.0)*np.pi*R_kern**3
            self.fw[:] = 4.0/3.0 * np.pi * R_kern**3 * \
                (spherical_jn(0, self.k_sin_R) + spherical_jn(2, self.k_sin_R))
        elif self.wf_type == WeightFunctionType.NORMTHETA:
            self.w_conv_steady = 1.0
            self.fw[:] = spherical_jn(0, self.k_sin_R) + spherical_jn(2, self.k_sin_R)
        elif self.wf_type == WeightFunctionType.DELTA:
            # The convolution of the fourier weights for steady profile
            self.w_conv_steady = 4.0*np.pi*R_kern**2
            self.fw[:] = 4.0 * np.pi * R_kern**2 * spherical_jn(0, self.k_sin_R)
        elif self.wf_type == WeightFunctionType.DELTAVEC:
            # The convolution of the fourier weights for steady profile
            self.w_conv_steady = 0.0
            #self.k_sin *
            self.fw[:] = 4.0/3.0 * np.pi * R_kern**3 * \
                (spherical_jn(0, self.k_sin_R) + spherical_jn(2, self.k_sin_R))


    def generate_polar_fourier_weights(self, grid, R):
        """
        """
        if self.wf_type == WeightFunctionType.THETA:
            pass
        elif self.wf_type == WeightFunctionType.NORMTHETA:
            pass
        elif self.wf_type == WeightFunctionType.DELTA:
            pass
        elif self.wf_type == WeightFunctionType.DELTAVEC:
            pass

    def convolve_densities(self, rho_inf: float, frho_delta: np.ndarray, weighted_density: np.ndarray):
        """

        Args:
            densities:
            rho (np.ndarray): Density profile

        """
        if self.convolve:
            if self.geometry == Geometry.PLANAR:
                self.planar_convolution(rho_inf, frho_delta, weighted_density)
            elif self.geometry == Geometry.SPHERICAL:
                self.spherical_convolution(rho_inf, frho_delta, weighted_density)
            elif self.geometry == Geometry.POLAR:
                self.polar_convolution(rho_inf, frho_delta, weighted_density)

    def update_dependencies(self, weighted_densities):
        """

        Args:
            densities:
            rho (np.ndarray): Density profile

        """
        if not self.convolve:
            weighted_densities[self.alias][:] = self.prefactor_evaluated*weighted_densities[self.calc_from][:]

    def planar_convolution(self, rho_inf: float, frho_delta: np.ndarray, weighted_density: np.ndarray):
        """

        Args:
            densities:
            rho (np.ndarray): Density profile

        """
        if self.wf_type == WeightFunctionType.DELTAVEC:
            # Sine transform
            # Fourier transfor only the rho_delta (the other term is done analytically)
            #self.frho_delta[:] = dct(self.rho_delta, type=2)
            frho_delta_V = np.roll(frho_delta.copy(), -1) # Fourier transform of density profile for `k_sin`
            frho_delta_V[-1] = 0                          # this information is lost but usually not important
            # Vector 2d weighted density (Sine transform)
            self.fn[:] = frho_delta_V[:] * self.fw[:]
            weighted_density[:] = idst(self.fn, type=2)
        else:
            # Cosine transform
            self.fn[:] = frho_delta[:] * self.fw[:]
            weighted_density[:] = idct(self.fn, type=2) + rho_inf*self.w_conv_steady

    def planar_convolution_differentials(self, diff: np.ndarray, diff_conv: np.ndarray):
        """

        Args:
            densities:
            rho (np.ndarray): Density profile

        """
        d_inf = diff[-1]
        d_delta = np.zeros_like(diff)
        d_delta[:] = diff[:] - d_inf
        if self.wf_type == WeightFunctionType.DELTAVEC:
            fd_delta = dst(d_delta, type=2)       # The vector valued function is odd
            self.fn[:] = fd_delta[:] * self.fw[:]
            # We must roll the vector to conform with the cosine transform
            self.fn = np.roll(self.fn, 1)
            self.fn[0] = 0
        else:
            fd_delta = dct(d_delta, type=2)
            self.fn[:] = fd_delta[:] * self.fw[:]

        diff_conv[:] = idct(self.fn, type=2) + d_inf*self.w_conv_steady


    def spherical_convolution(self, rho_inf: float, frho_delta: np.ndarray, weighted_density: np.ndarray):
        """

        Args:
            densities:
            rho (np.ndarray): Density profile

        """
        if self.wf_type == WeightFunctionType.DELTAVEC:
            # Vector 2d weighted density (Spherical geometry)
            weighted_density[:] = (self.one_div_r**2)*idst(frho_delta[:] * self.fw[:], type=2)
            # Intermediate calculations for the intermediate term (Spherical geometry)
            self.fn[:] = self.k_sin*self.fw*frho_delta
            self.fn[:] = np.roll(self.fn, 1)
            self.fn[0] = 0
            weighted_density[:] -= self.one_div_r*idct(self.fn, type=2)
        else:
            self.fn[:] = frho_delta[:] * self.fw[:]
            weighted_density[:] = self.one_div_r*idst(self.fn, type=2) + rho_inf*self.w_conv_steady

    def spherical_convolution_differentials(self, diff: np.ndarray, diff_conv: np.ndarray):
        """

        Args:
            densities:
            rho (np.ndarray): Density profile

        """
        d_inf = diff[-1]
        d_delta = np.zeros_like(diff)
        d_delta[:] = diff[:] - d_inf
        if self.wf_type == WeightFunctionType.DELTAVEC:
            # Next handle the vector term in spherical coordinates (has two terms)
            fd_term1 = -1.0*dst(d_delta, type=2)/(self.k_sin)
            fd_term2 = dct(self.r*d_delta, type=2)
            fd_term2 = np.roll(fd_term2, -1)
            fd_term2[-1] = 0.0
            self.fn[:] = -1.0*(fd_term1[:] + fd_term2[:])*self.fw[:]*self.k_sin[:]
        else:
            fd_delta = dst(d_delta*self.r, type=2)
            self.fn[:] = fd_delta[:] * self.fw[:]

        # Transform from Fourier space to real space
        diff_conv[:] = self.one_div_r*idst(self.fn, type=2) + d_inf*self.w_conv_steady

    def polar_convolution(self, rho_inf: float, frho_delta: np.ndarray, weighted_density: np.ndarray):
        """

        Args:
            densities:
            rho (np.ndarray): Density profile

        """
        if self.wf_type == WeightFunctionType.DELTAVEC:
            pass
        else:
            pass

    def convolve_differentials(self, diff: np.ndarray, conv_diff: np.ndarray):
        """

        Args:
            densities:
            rho (np.ndarray): Density profile

        """
        if self.convolve:
            if self.geometry == Geometry.PLANAR:
                self.planar_convolution_differentials(diff, conv_diff)
            elif self.geometry == Geometry.SPHERICAL:
                self.spherical_convolution_differentials(diff, conv_diff)
            elif self.geometry == Geometry.POLAR:
                self.polar_convolution_differentials(diff, conv_diff)

class WeightFunctions(object):

    def __init__(self):
        """
        """
        self.wfs = {}
        self.fmt_aliases = []

    @staticmethod
    def Copy(Other):
        """Method used to duplicate WeightFunctions for different components

        Args:
        Other (WeightFunctions): Other instance of WeightFunctions
        """
        wfs = WeightFunctions()
        wfs.fmt_aliases = Other.fmt_aliases
        for wf in Other.wfs:
            wfs.wfs[wf] = WeightFunction.Copy(Other.wfs[wf])
        return wfs

    def __getitem__(self, wf):
        return self.wfs[wf]

    def items(self):
        return self.wfs.items()

    def __contains__(self, key):
        return key in self.wfs

    def __iter__(self):
        return iter(self.wfs.keys())

    def append(self, wf):
        self.wfs[wf.alias] = wf

    def add_fmt_weights(self):
        self.wfs["w0"] = WeightFunction(WeightFunctionType.DELTA,
                                        kernel_radius=1.0,
                                        alias = "w0",
                                        prefactor = "1.0/(4.0*pi*R**2)",
                                        convolve=False,
                                        calc_from="w2")
        self.fmt_aliases.append("w0")
        self.wfs["w1"] = WeightFunction(WeightFunctionType.DELTA,
                                        kernel_radius=1.0,
                                        alias = "w1",
                                        prefactor = "1.0/(4.0*pi*R)",
                                        convolve=False,
                                        calc_from="w2")
        self.fmt_aliases.append("w1")
        self.wfs["w2"] = WeightFunction(WeightFunctionType.DELTA,
                                        kernel_radius=1.0,
                                        alias = "w2",
                                        prefactor = "1.0",
                                        convolve=True)
        self.fmt_aliases.append("w2")
        self.wfs["w3"] = WeightFunction(WeightFunctionType.THETA,
                                        kernel_radius=1.0,
                                        alias = "w3",
                                        prefactor = "1.0",
                                        convolve=True)
        self.fmt_aliases.append("w3")
        self.wfs["wv1"] = WeightFunction(WeightFunctionType.DELTAVEC,
                                         kernel_radius=1.0,
                                         alias = "wv1",
                                         prefactor = "1.0/(4.0*pi*R)",
                                         convolve=False,
                                         calc_from="wv2")
        self.fmt_aliases.append("wv1")
        self.wfs["wv2"] = WeightFunction(WeightFunctionType.DELTAVEC,
                                         kernel_radius=1.0,
                                         alias = "wv2",
                                         prefactor = "1.0",
                                         convolve=True)
        self.fmt_aliases.append("wv2")

    def add_norm_theta_weight(self, alias, kernel_radius):
        self.wfs[alias] = WeightFunction(WeightFunctionType.NORMTHETA,
                                         kernel_radius=kernel_radius,
                                         alias = alias,
                                         prefactor = "1.0", # Not important. Will be calculated.
                                         convolve=True)

    def get_correlation_factor(self, label, R):
        """
        """
        corr_fac = 0.0
        for wf in self.wfs:
            if self.wfs[wf].convolve and wf == label:
                corr_fac += 1.0
            elif self.wfs[wf].calc_from == label:
                print("R",R)
                corr_fac += self.wfs[wf].prefactor(R)
        return corr_fac


def get_functional(N, T, functional="Rosenfeld", R=np.array([0.5]), thermopack=None):
    """
    Return functional class based on functional name.

    Args:
        N (int): Grid size
        T (float): Reduced temperature
        functional (str): Name of functional
        R (ndarray): Particle radius for all components
        thermopack (thermo): Thermopack object
    """
    functional_name = functional.upper().strip(" -")
    if functional_name in ("ROSENFELD", "RF"):
        func = Rosenfeld(N=N, R=R)
    elif functional_name in ("WHITEBEAR", "WB"):
        func = Whitebear(N=N, R=R)
    elif functional_name in ("WHITEBEARMARKII", "WHITEBEARII", "WBII"):
        func = WhitebearMarkII(N=N, R=R)
    elif functional_name in ("PC-SAFT", "PCSAFT"):
        func = pc_saft(N=N, pcs=thermopack, T_red=T)
    else:
        raise ValueError("Unknown functional: " + functional)

    return func


class bulk_weighted_densities:
    """
    Utility class for calculating bulk states.
    """

    def __init__(self, rho_b, R):
        """

        Args:
            rho_b (ndarray): Bulk densities
            R (ndarray): Particle radius for all components
        """
        self.rho_i = np.zeros_like(rho_b)
        self.rho_i[:] = rho_b[:]
        self.n = np.zeros(4)
        self.n[0] = np.sum(rho_b)
        self.n[1] = np.sum(R * rho_b)
        self.n[2] = 4*np.pi*np.sum(R ** 2 * rho_b)
        self.n[3] = 4 * np.pi * np.sum(R ** 3 * rho_b) / 3
        self.dndrho = np.zeros((4, np.shape(rho_b)[0]))
        self.dndrho[0, :] = 1.0
        self.dndrho[1, :] = R
        self.dndrho[2, :] = 4*np.pi*R**2
        self.dndrho[3, :] = 4*np.pi*R**3/3

    def print(self):
        print("Bulk weighted densities:")
        print("n_0: ", self.n[0])
        print("n_1: ", self.n[1])
        print("n_2: ", self.n[2])
        print("n_3: ", self.n[3])


class Rosenfeld:
    """
    Rosenfeld, Yaakov
    Free-energy model for the inhomogeneous hard-sphere fluid mixture andl
    density-functional theory of freezing.
    Phys. Rev. Lett. 1989, 63(9):980-983
    doi:10.1103/PhysRevLett.63.980
    """

    def __init__(self, N, R=np.array([0.5])):
        """

        Args:
            N (integer): Grid size
            R (ndarray): Particle radius for all components
        """
        self.name = "Rosenfeld"
        self.short_name = "RF"
        self.R = R
        self.nc = np.shape(R)[0]
        self.n_grid = N
        # Allocate arrays for differentials
        self.d0 = np.zeros(N)
        self.d1 = np.zeros(N)
        self.d2 = np.zeros(N)
        self.d3 = np.zeros(N)
        self.d1v = np.zeros(N)
        self.d2v = np.zeros(N)
        self.d2eff = np.zeros(N)
        self.d2veff = np.zeros(N)

        # Set up FMT weights
        self.wf = WeightFunctions()
        self.wf.add_fmt_weights()

        # Differentials
        self.diff = {}
        for wf in self.wf.wfs:
            alias = self.wf.wfs[wf].alias
            if "v" in alias:
                if "1" in alias:
                    self.diff[alias] = self.d1v
                else:
                    self.diff[alias] = self.d2v
            elif "0" in alias:
                self.diff[alias] = self.d0
            elif "1" in alias:
                self.diff[alias] = self.d1
            elif "2" in alias:
                self.diff[alias] = self.d2
            elif "3" in alias:
                self.diff[alias] = self.d3

    def excess_free_energy(self, dens):
        """
        Calculates the excess HS Helmholtz free energy from the weighted densities

        Args:
            dens (array_like): Weighted densities

        Returns:
            array_like: Excess HS Helmholtz free energy ()

        """
        f = np.zeros(dens.n_grid)
        f[:] = -dens.n0[:] * dens.logn3neg[:] + \
            (dens.n1[:] * dens.n2[:] - dens.n1v[:] * dens.n2v[:]) / dens.n3neg[:] + \
            ((dens.n2[:] ** 3) - 3.0 * dens.n2[:] * dens.n2v[:]
             ** 2) / (24.0 * np.pi * dens.n3neg[:] ** 2)

        return f

    def bulk_compressibility(self, rho_b):
        """
        Calculates the Percus-Yevick HS compressibility from the
        packing fraction. Multiply by rho*kB*T to get pressure.

        Args:
            rho_b (ndarray): Bulk densities

        Returns:
            float: compressibility
        """
        bd = bulk_weighted_densities(rho_b, self.R)
        phi, dphidn = self.bulk_functional_with_differentials(
            bd, only_hs_system=True)
        beta_p_ex = - phi + np.sum(dphidn[:4] * bd.n)
        beta_p_id = bd.n[0]
        z = (beta_p_id + beta_p_ex)/bd.n[0]
        return z

    def bulk_excess_chemical_potential(self, rho_b):
        """
        Calculates the reduced HS excess chemical potential from the bulk
        packing fraction.

        Args:
        rho_b (ndarray): Bulk densities

        Returns:
        float: Excess reduced HS chemical potential ()

        """
        bd = bulk_weighted_densities(rho_b, self.R)
        phi, dphidn = self.bulk_functional_with_differentials(
            bd, only_hs_system=True)
        mu_ex = np.zeros(self.nc)
        for i in range(self.nc):
            mu_ex[i] = np.sum(dphidn[:4] * bd.dndrho[:, i])

        return mu_ex

    def bulk_functional_with_differentials(self, bd, only_hs_system=False):
        """
        Calculates the functional differentials wrpt. the weighted densities
        in the bulk phase.

        Args:
        bd (bulk_weighted_densities): bulk_weighted_densities
        only_hs_system (bool): Only calculate for hs-system

        """
        n3neg = 1.0-bd.n[3]
        d0 = -np.log(n3neg)
        d1 = bd.n[2] / n3neg
        d2 = bd.n[1] / n3neg + bd.n[2] ** 2 / (8 * np.pi * n3neg ** 2)
        d3 = bd.n[0] / n3neg + bd.n[1] * bd.n[2] / n3neg ** 2 \
            + bd.n[2] ** 3 / (12 * np.pi * n3neg ** 3)
        dphidn = np.array([d0, d1, d2, d3])
        phi = d0 * bd.n[0] + d1 * bd.n[1] + \
            bd.n[2] ** 3 / (24 * np.pi * n3neg ** 2)
        return phi, dphidn

    def differentials(self, dens):
        """
        Calculates the functional differentials wrpt. the weighted densities.

        Args:
        dens (array_like): weighted densities
        diff (array_like): Functional differentials

        """
        self.d0[:] = -np.log(dens.n3neg[:])
        self.d1[:] = dens.n2[:] / dens.n3neg[:]
        self.d2[:] = dens.n1[:] / dens.n3neg[:] + \
            (dens.n2[:] ** 2 - dens.n2v2[:]) / (8 * np.pi * dens.n3neg2[:])
        self.d3[:] = dens.n0[:] / dens.n3neg[:] + (dens.n1[:] * dens.n2[:] - dens.n1v[:] * dens.n2v[:]) / \
            dens.n3neg2[:] + (dens.n2[:] ** 3 - 3 * dens.n2[:] * dens.n2v2[:]) / \
            (12 * np.pi * dens.n3neg[:] ** 3)
        self.d1v[:] = -dens.n2v[:] / dens.n3neg[:]
        self.d2v[:] = -(dens.n1v[:] / dens.n3neg[:] + dens.n2[:]
                        * dens.n2v[:] / (4 * np.pi * dens.n3neg2[:]))

        # Combining differentials
        self.combine_differentials()

    def combine_differentials(self):
        """
        Combining differentials to reduce number of convolution integrals
        """
        self.d2eff[:] = self.d0[:] / (4 * np.pi * self.R ** 2) + \
            self.d1[:] / (4 * np.pi * self.R) + self.d2[:]
        self.d2veff[:] = self.d1v[:] / (4 * np.pi * self.R) + self.d2v[:]

    def get_differential(self, i):
        """
        Get differential number i
        """
        if i == 0:
            d = self.d0
        elif i == 1:
            d = self.d1
        elif i == 2:
            d = self.d2
        elif i == 3:
            d = self.d3
        elif i == 4:
            d = self.d1v
        elif i == 5:
            d = self.d2v
        else:
            raise ValueError("get_differential: Index out of bounds")
        return d

    def get_bulk_correlation(self, rho_b, only_hs_system=False):
        """
        Intended only for debugging
        Args:
            rho_b (np.ndarray): Bulk densities
            only_hs_system (bool): Only calculate for hs-system

        Return:
            corr (np.ndarray): One particle correlation function
        """
        bd = bulk_weighted_densities(rho_b, self.R)
        _, dphidn = self.bulk_functional_with_differentials(
            bd, only_hs_system=only_hs_system)
        corr = np.zeros(self.nc)
        for i in range(self.nc):
            corr[i] = dphidn[0] + \
                self.R[i] * dphidn[1] + \
                (4 * np.pi * self.R[i] ** 2) * dphidn[2] + \
                4 * np.pi * self.R[i] ** 3 * dphidn[3] / 3
            if np.shape(dphidn)[0] > 4:
                corr[i] += dphidn[4+i]

        corr = -corr
        return corr

    def test_differentials(self, dens0):
        """

        Args:
            dens0 (weighted_densities_1D): Weighted densities

        """
        print("Testing functional " + self.name)
        self.differentials(dens0)
        #F0 = self.excess_free_energy(dens0)
        eps = 1.0e-5
        ni0 = np.zeros_like(dens0.n0)
        dni = np.zeros_like(dens0.n0)
        for i in range(dens0.n_max_test):
            ni0[:] = dens0.get_density(i)
            dni[:] = eps * ni0[:]
            dens0.set_density(i, ni0 - dni)
            dens0.update_utility_variables()
            F1 = self.excess_free_energy(dens0)
            dens0.set_density(i, ni0 + dni)
            dens0.update_utility_variables()
            F2 = self.excess_free_energy(dens0)
            dens0.set_density(i, ni0)
            dens0.update_utility_variables()
            dFdn_num = (F2 - F1) / (2 * dni)
            print("Differential: ", i, dFdn_num, self.get_differential(i), (dFdn_num -self.get_differential(i))/self.get_differential(i))

    def test_bulk_differentials(self, rho_b):
        """

        Args:
            rho_b (np.ndarray): Bulk densities

        """
        print("Testing functional " + self.name)
        bd0 = bulk_weighted_densities(rho_b, self.R)
        phi, dphidn = self.bulk_functional_with_differentials(bd0)

        print("HS functional differentials:")
        for i in range(4):
            bd = bulk_weighted_densities(rho_b, self.R)
            eps = 1.0e-5 * bd.n[i]
            bd.n[i] += eps
            phi2, dphidn2 = self.bulk_functional_with_differentials(bd)
            bd = bulk_weighted_densities(rho_b, self.R)
            eps = 1.0e-5 * bd.n[i]
            bd.n[i] -= eps
            phi1, dphidn1 = self.bulk_functional_with_differentials(bd)
            dphidn_num = (phi2 - phi1) / (2 * eps)
            print("Differential: ", i, dphidn_num, dphidn[i])

        mu_ex = self.bulk_excess_chemical_potential(rho_b)
        rho_b_local = np.zeros_like(rho_b)
        print("Functional differentials:")
        for i in range(self.nc):
            eps = 1.0e-5 * rho_b[i]
            rho_b_local[:] = rho_b[:]
            rho_b_local[i] += eps
            bd = bulk_weighted_densities(rho_b_local, self.R)
            phi2, dphidn1 = self.bulk_functional_with_differentials(bd)
            phi2_hs, _ = self.bulk_functional_with_differentials(
                bd, only_hs_system=True)
            rho_b_local[:] = rho_b[:]
            rho_b_local[i] -= eps
            bd = bulk_weighted_densities(rho_b_local, self.R)
            phi1, dphidn1 = self.bulk_functional_with_differentials(bd)
            phi1_hs, _ = self.bulk_functional_with_differentials(
                bd, only_hs_system=True)
            dphidrho_num_no_hs = (phi2 - phi2_hs - phi1 + phi1_hs) / (2 * eps)
            dphidrho_num = (phi2 - phi1) / (2 * eps)
            if np.shape(dphidn)[0] > 4:
                print("Differential: ", 4+i, dphidrho_num_no_hs, dphidn[4+i], (dphidrho_num_no_hs - dphidn[4+i])/dphidn[4+i])
            print("Chemical potential comp.: ", i, dphidrho_num, mu_ex[i])


class Whitebear(Rosenfeld):
    """
    R. Roth, R. Evans, A. Lang and G. Kahl
    Fundamental measure theory for hard-sphere mixtures revisited: the White Bear version
    Journal of Physics: Condensed Matter
    2002, 14(46):12063-12078
    doi: 10.1088/0953-8984/14/46/313

    In the bulk phase the functional reduces to the Boublik and
    Mansoori, Carnahan, Starling, and Leland (BMCSL) EOS.
    T. Boublik, doi: 10/bjgkjg
    G. A. Mansoori, N. F. Carnahan, K. E. Starling, T. Leland, doi: 10/dkfhh7

    """

    def __init__(self, N, R=np.array([0.5])):
        """

        Args:
            N (integer): Grid size
            R (ndarray): Particle radius for all components
        """
        Rosenfeld.__init__(self, N, R)
        self.name = "White Bear"
        self.short_name = "WB"
        self.numerator = None
        self.denumerator = None

    def excess_free_energy(self, dens):
        """
        Calculates the excess HS Helmholtz free energy from the weighted densities

        Args:
        dens (array_like): Weighted densities

        Returns:
        array_like: Excess HS Helmholtz free energy ()

        """
        # Avoid dividing with zero value of n3 in boundary grid points
        pn3m = dens.n3 > 0.0  # Positive value n3 mask
        f = np.zeros(dens.n_grid)
        f[pn3m] = -dens.n0[pn3m] * dens.logn3neg[pn3m] + \
            (dens.n1[pn3m] * dens.n2[pn3m] - dens.n1v[pn3m] * dens.n2v[pn3m]) / dens.n3neg[pn3m] + \
            ((dens.n2[pn3m] ** 3) - 3.0 * dens.n2[pn3m] * dens.n2v[pn3m] ** 2) * \
            (dens.n3[pn3m] + dens.n3neg2[pn3m] * dens.logn3neg[pn3m]) / \
            (36.0 * np.pi * dens.n3[pn3m] ** 2 * dens.n3neg2[pn3m])

        return f

    def bulk_functional_with_differentials(self, bd, only_hs_system=False):
        """
        Calculates the functional differentials wrpt. the weighted densities
        in the bulk phase.

        Args:
        bd (bulk_weighted_densities): bulk_weighted_densities
        only_hs_system (bool): Only calculate for hs-system

        """
        n3neg = 1.0-bd.n[3]
        numerator = bd.n[3] + n3neg ** 2 * np.log(n3neg)
        denumerator = 36.0 * np.pi * bd.n[3] ** 2 * n3neg ** 2
        d0 = -np.log(n3neg)
        d1 = bd.n[2] / n3neg
        d2 = bd.n[1] / n3neg + 3 * bd.n[2] ** 2 * numerator / denumerator
        d3 = bd.n[0] / n3neg + \
            bd.n[1] * bd.n[2] / n3neg ** 2 + \
            bd.n[2] ** 3 * \
            ((bd.n[3] * (5 - bd.n[3]) - 2) /
             (denumerator * n3neg) - np.log(n3neg) /
             (18 * np.pi * bd.n[3] ** 3))

        dphidn = np.array([d0, d1, d2, d3])
        phi = d0 * bd.n[0] + d1 * bd.n[1] + \
            bd.n[2] ** 3 * numerator / denumerator
        return phi, dphidn

    def differentials(self, dens):
        """
        Calculates the functional differentials wrpt. the weighted densities

        Args:
        dens (array_like): weighted densities

        """
        if self.numerator is None or np.shape(self.numerator) != np.shape(dens.n0):
            self.numerator = np.zeros_like(dens.n0)
        self.numerator[:] = dens.n3[:] + dens.n3neg2[:] * dens.logn3neg[:]
        if self.denumerator is None or np.shape(self.denumerator) != np.shape(dens.n0):
            self.denumerator = np.zeros_like(dens.n0)
        self.denumerator[:] = (36.0 * np.pi * dens.n3[:] ** 2 * dens.n3neg2[:])

        # Avoid dividing with zero value of n3 in boundary grid points
        pn3m = dens.n3 > 0.0  # Positive value n3 mask
        non_pn3m = np.invert(pn3m)  # Mask for zero and negative value of n3

        self.d0[pn3m] = -dens.logn3neg[pn3m]
        self.d1[pn3m] = dens.n2[pn3m] / dens.n3neg[pn3m]
        self.d2[pn3m] = dens.n1[pn3m] / dens.n3neg[pn3m] + 3 * (dens.n2[pn3m] ** 2 - dens.n2v2[pn3m]) * \
            self.numerator[pn3m] / self.denumerator[pn3m]
        self.d3[pn3m] = dens.n0[pn3m] / dens.n3neg[pn3m] + \
            (dens.n1[pn3m] * dens.n2[pn3m] - dens.n1v[pn3m] * dens.n2v[pn3m]) / dens.n3neg2[pn3m] + \
            (dens.n2[pn3m] ** 3 - 3 * dens.n2[pn3m] * dens.n2v2[pn3m]) * \
            ((dens.n3[pn3m] * (5 - dens.n3[pn3m]) - 2) /
             (self.denumerator[pn3m] * dens.n3neg[pn3m]) - dens.logn3neg[pn3m] / (
                18 * np.pi * dens.n3[pn3m] ** 3))
        self.d1v[pn3m] = -dens.n2v[pn3m] / dens.n3neg[pn3m]
        self.d2v[pn3m] = -dens.n1v[pn3m] / dens.n3neg[pn3m] - 6 * dens.n2[pn3m] * dens.n2v[pn3m] * \
            self.numerator[pn3m] / self.denumerator[pn3m]

        # Combining differentials
        self.combine_differentials()

        # Set non positive n3 grid points to zero
        self.d3[non_pn3m] = 0.0
        self.d2eff[non_pn3m] = 0.0
        self.d2veff[non_pn3m] = 0.0


class WhitebearMarkII(Whitebear):
    """
    Hendrik Hansen-Goos and Roland Roth
    Density functional theory for hard-sphere mixtures:
    the White Bear version mark II.
    Journal of Physics: Condensed Matter
    2006, 18(37): 8413-8425
    doi: 10.1088/0953-8984/18/37/002
    """

    def __init__(self, N, R=np.array([0.5])):
        """

        Args:
            R (ndarray): Radius of particles
        """
        Whitebear.__init__(self, N, R)
        self.name = "White Bear Mark II"
        self.short_name = "WBII"
        self.phi2_div3 = None
        self.dphi2dn3_div3 = None
        self.phi3_div3 = None
        self.dphi3dn3_div3 = None

    def update_phi2_and_phi3(self, dens, mask=None):
        """
        Calculates function Phi2 from n3

        Args:
        dens (array_like): Weighted densities
        mask (np.ndarray boolean): Mask for updating phi2 and phi3
        """
        if self.phi2_div3 is None or np.shape(self.phi2_div3) != np.shape(dens.n3):
            self.phi2_div3 = np.zeros_like(dens.n3)
        if self.dphi2dn3_div3 is None or np.shape(self.dphi2dn3_div3) != np.shape(dens.n3):
            self.dphi2dn3_div3 = np.zeros_like(dens.n3)
        if self.phi3_div3 is None or np.shape(self.phi3_div3) != np.shape(dens.n3):
            self.phi3_div3 = np.zeros_like(dens.n3)
        if self.dphi3dn3_div3 is None or np.shape(self.dphi3dn3_div3) != np.shape(dens.n3):
            self.dphi3dn3_div3 = np.zeros_like(dens.n3)
        if mask is None:
            mask = np.full(dens.n_grid, True, dtype=bool)
        prefac = 1.0 / 3.0
        self.phi2_div3[mask] = prefac * (2 - dens.n3[mask] + 2 *
                                         dens.n3neg[mask] * dens.logn3neg[mask] / dens.n3[mask])
        self.dphi2dn3_div3[mask] = prefac * \
            (- 1 - 2 / dens.n3[mask] - 2 *
             dens.logn3neg[mask] / dens.n32[mask])
        self.phi3_div3[mask] = prefac * (
            2 * dens.n3[mask] - 3 * dens.n32[mask] + 2 * dens.n3[mask] * dens.n32[mask] +
            2 * dens.n3neg2[mask] * dens.logn3neg[mask]) / dens.n32[mask]
        self.dphi3dn3_div3[mask] = prefac * (
            - 4 * dens.logn3neg[mask] * dens.n3neg[mask] /
            (dens.n32[mask] * dens.n3[mask])
            - 4 / dens.n32[mask] + 2 / dens.n3[mask] + 2)

    def excess_free_energy(self, dens):
        """
        Calculates the excess HS Helmholtz free energy from the weighted densities

        Args:
        dens (array_like): Weighted densities

        Returns:
        array_like: Excess HS Helmholtz free energy ()

        """
        # Avoid dividing with zero value of n3 in boundary grid points
        pn3m = dens.n3 > 0.0  # Positive value n3 mask
        self.update_phi2_and_phi3(dens, pn3m)
        f = np.zeros(dens.n_grid)
        f[pn3m] = -dens.n0[pn3m] * dens.logn3neg[pn3m] + \
            (dens.n1[pn3m] * dens.n2[pn3m] - dens.n1v[pn3m] * dens.n2v[pn3m]) * \
            (1 + self.phi2_div3) / (dens.n3neg[pn3m]) + \
            ((dens.n2[pn3m] ** 3) - 3.0 * dens.n2[pn3m] * dens.n2v[pn3m] ** 2) * \
            (1 - self.phi3_div3[pn3m]) / (24.0 * np.pi * dens.n3neg2[pn3m])

        return f

    def calc_phi2_and_phi3_bulk(self, bd):
        """
        Calculates function  Phi2 and Phi3 from n3

        Args:
        dens (bulk_weighted_densities): Weighted densities
        """

        prefac = 1.0 / 3.0
        n3neg = 1.0-bd.n[3]
        logn3neg = np.log(n3neg)
        phi2_div3 = prefac * (2 - bd.n[3] + 2 *
                              n3neg * logn3neg / bd.n[3])
        dphi2dn3_div3 = prefac * \
            (- 1 - 2 / bd.n[3] - 2 * logn3neg / bd.n[3] ** 2)
        phi3_div3 = prefac * (
            2 * bd.n[3] - 3 * bd.n[3] ** 2 + 2 * bd.n[3] * bd.n[3] ** 2 +
            2 * n3neg ** 2 * logn3neg) / bd.n[3] ** 2
        dphi3dn3_div3 = prefac * \
            (- 4 * logn3neg * n3neg /
             (bd.n[3] ** 2 * bd.n[3]) - 4 / bd.n[3] ** 2 + 2 / bd.n[3] + 2)
        return phi2_div3, dphi2dn3_div3, phi3_div3, dphi3dn3_div3

    def bulk_functional_with_differentials(self, bd, only_hs_system=False):
        """
        Calculates the functional differentials wrpt. the weighted densities
        in the bulk phase.

        Args:
        bd (bulk_weighted_densities): bulk_weighted_densities
        only_hs_system (bool): Only calculate for hs-system

        """
        phi2_div3, dphi2dn3_div3, phi3_div3, dphi3dn3_div3 = \
            self.calc_phi2_and_phi3_bulk(bd)
        n3neg = 1.0-bd.n[3]
        numerator = 1 - phi3_div3
        denumerator = 24.0 * np.pi * n3neg ** 2
        d0 = -np.log(n3neg)
        d1 = bd.n[2] * (1 + phi2_div3) / n3neg
        d2 = bd.n[1] * (1 + phi2_div3) / n3neg + 3 * \
            bd.n[2] ** 2 * numerator / denumerator
        d3 = bd.n[0] / n3neg + \
            bd.n[1] * bd.n[2] * \
            ((1 + phi2_div3) / n3neg ** 2 +
             dphi2dn3_div3 / n3neg) + \
            bd.n[2] ** 3 / denumerator * \
            (-dphi3dn3_div3 + 2 *
             numerator / n3neg)

        dphidn = np.array([d0, d1, d2, d3])
        phi = d0 * bd.n[0] + d1 * bd.n[1] + \
            bd.n[2] ** 3 * numerator / denumerator
        return phi, dphidn

    def differentials(self, dens):
        """
        Calculates the functional differentials wrpt. the weighted densities

        Args:
        dens (array_like): weighted densities

        """
        # Avoid dividing with zero value of n3 in boundary grid points
        pn3m = dens.n3 > 0.0  # Positive value n3 mask
        non_pn3m = np.invert(pn3m)  # Mask for zero and negative value of n3
        self.update_phi2_and_phi3(dens, pn3m)
        if self.numerator is None or np.shape(self.numerator) != np.shape(dens.n0):
            self.numerator = np.zeros_like(dens.n0)
        self.numerator[:] = 1 - self.phi3_div3[:]
        if self.denumerator is None or np.shape(self.denumerator) != np.shape(dens.n0):
            self.denumerator = np.zeros_like(dens.n0)
        self.denumerator[:] = (24.0 * np.pi * dens.n3neg2[:])

        self.d0[pn3m] = -dens.logn3neg[pn3m]
        self.d1[pn3m] = dens.n2[pn3m] * \
            (1 + self.phi2_div3[pn3m]) / dens.n3neg[pn3m]
        self.d2[pn3m] = dens.n1[pn3m] * (1 + self.phi2_div3[pn3m]) / dens.n3neg[pn3m] + 3 * (
            dens.n2[pn3m] ** 2 - dens.n2v2[pn3m]) * self.numerator[pn3m] / self.denumerator[pn3m]
        self.d3[pn3m] = dens.n0[pn3m] / dens.n3neg[pn3m] + \
            (dens.n1[pn3m] * dens.n2[pn3m] - dens.n1v[pn3m] * dens.n2v[pn3m]) * \
            ((1 + self.phi2_div3[pn3m]) / dens.n3neg2[pn3m] +
             self.dphi2dn3_div3[pn3m] / dens.n3neg[pn3m]) + \
            (dens.n2[pn3m] ** 3 - 3 * dens.n2[pn3m] * dens.n2v2[pn3m]) / self.denumerator[pn3m] * \
            (-self.dphi3dn3_div3[pn3m] + 2 *
             self.numerator[pn3m] / dens.n3neg[pn3m])
        self.d1v[pn3m] = -dens.n2v[pn3m] * \
            (1 + self.phi2_div3[pn3m]) / dens.n3neg[pn3m]
        self.d2v[pn3m] = -dens.n1v[pn3m] * (1 + self.phi2_div3[pn3m]) / dens.n3neg[pn3m] \
            - 6 * dens.n2[pn3m] * dens.n2v[pn3m] * \
            self.numerator[pn3m] / self.denumerator[pn3m]

        # Combining differentials
        self.combine_differentials()

        # Set non positive n3 grid points to zero
        self.d3[non_pn3m] = 0.0
        self.d2eff[non_pn3m] = 0.0
        self.d2veff[non_pn3m] = 0.0


class pc_saft(Whitebear):
    """

    """

    def __init__(self, N, pcs: pcsaft, T_red, phi_disp=1.3862):
        """

        Args:
            pcs (pcsaft): Thermopack object
            T_red (float): Reduced temperature
            R (ndarray): Particle radius for all components
        """
        self.thermo = pcs
        self.T_red = T_red
        self.T = self.T_red * self.thermo.eps_div_kb[0]
        self.d_hs, self.d_T_hs = pcs.hard_sphere_diameters(self.T)
        R = np.zeros(pcs.nc)
        R[:] = 0.5*self.d_hs[:]/self.d_hs[0]
        Whitebear.__init__(self, N, R)
        self.name = "PC-SAFT"
        self.short_name = "PC"
        # Add normalized theta weight
        self.mu_disp = np.zeros((N, pcs.nc))
        self.disp_name = "w_disp"
        self.wf.add_norm_theta_weight(self.disp_name, kernel_radius=phi_disp)
        self.diff[self.disp_name] = self.mu_disp

    def excess_free_energy(self, dens):
        """
        Calculates the excess HS Helmholtz free energy from the weighted densities

        Args:
        dens (array_like): Weighted densities

        Returns:
        array_like: Excess HS Helmholtz free energy ()

        """
        f = Whitebear.excess_free_energy(self, dens)
        rho_thermo = np.zeros(self.nc)
        V = 1.0
        for i in range(len(f)):
            rho_thermo[:] = dens.rho_disp_array[:, i]
            rho_mix = np.sum(rho_thermo)
            rho_thermo *= 1.0/(NA*self.d_hs[0]**3)
            a, = self.thermo.a_dispersion(self.T, V, rho_thermo)
            f[i] += rho_mix*a

        return f

    def differentials(self, dens):
        """
        Calculates the functional differentials wrpt. the weighted densities

        Args:
        dens (array_like): weighted densities
        diff (array_like): Functional differentials

        """
        Whitebear.differentials(self, dens)

        # All densities must be positive
        prdm = dens.rho_disp > 0.0  # Positive rho_disp value mask
        for i in range(self.nc):
            np.logical_and(prdm, dens.rho_disp_array[:, i] > 0.0, out=prdm)
        # Mask for zero and negative value of rho_disp
        non_prdm = np.invert(prdm)
        rho_thermo = np.zeros(self.nc)
        V = 1.0
        for i in range(self.n_grid):
            if prdm[i]:
                rho_thermo[:] = dens.rho_disp_array[:, i]
                rho_thermo *= 1.0/(NA*self.d_hs[0]**3)
                a, a_n, = self.thermo.a_dispersion(
                    self.T, V, rho_thermo, a_n=True)
                self.mu_disp[i, :] = (a + rho_thermo[:]*a_n[:])
        self.mu_disp[non_prdm, :] = 0.0

    def bulk_compressibility(self, rho_b):
        """
        Calculates the PC-SAFT compressibility.
        Multiply by rho*kB*T to get pressure.

        Args:
            rho_b (ndarray): Bulk densities

        Returns:
            float: compressibility
        """
        z = Whitebear.bulk_compressibility(self, rho_b)
        # PC-SAFT contributions
        rho_thermo = np.zeros_like(rho_b)
        rho_thermo[:] = rho_b[:]
        rho_thermo *= 1.0/(NA*self.d_hs[0]**3)
        rho_mix = np.sum(rho_thermo)
        V = 1.0/rho_mix
        n = rho_thermo/rho_mix
        a, a_V, = self.thermo.a_dispersion(
            self.T, V, n, a_v=True)
        z_r = -a_V*V
        z += z_r
        return z

    def bulk_excess_chemical_potential(self, rho_b):
        """
        Calculates the reduced HS excess chemical potential from the bulk
        packing fraction.

        Args:
        rho_b (ndarray): Bulk densities

        Returns:
        float: Excess reduced HS chemical potential ()

        """
        mu_ex = Whitebear.bulk_excess_chemical_potential(self, rho_b)
        # PC-SAFT contributions
        rho_thermo = np.zeros_like(rho_b)
        rho_thermo[:] = rho_b[:]
        rho_thermo *= 1.0/(NA*self.d_hs[0]**3)
        rho_mix = np.sum(rho_thermo)
        V = 1.0
        n = rho_thermo
        a, a_n, = self.thermo.a_dispersion(
            self.T, V, n, a_n=True)
        a_n *= n
        a_n += a
        mu_ex += a_n
        return mu_ex

    def bulk_functional_with_differentials(self, bd, only_hs_system=False):
        """
        Calculates the functional differentials wrpt. the weighted densities
        in the bulk phase.

        Args:
        bd (bulk_weighted_densities): bulk_weighted_densities
        only_hs_system (bool): Only calculate for hs-system
        """
        phi, dphidn = Whitebear.bulk_functional_with_differentials(self, bd)
        if not only_hs_system:
            rho_vec = bd.rho_i
            rho_mix = np.sum(rho_vec)
            V = 1.0
            rho_thermo = np.zeros_like(rho_vec)
            rho_thermo[:] = rho_vec[:]/(NA*self.d_hs[0]**3)
            a, a_n, = self.thermo.a_dispersion(
                self.T, V, rho_thermo, a_n=True)
            phi += rho_mix*a
            dphidn_comb = np.zeros(4 + self.nc)
            dphidn_comb[:4] = dphidn
            dphidn_comb[4:] = a + rho_thermo[:]*a_n[:]
        else:
            dphidn_comb = dphidn
        return phi, dphidn_comb

    def get_differential(self, i):
        """
        Get differential number i
        """
        if i <= 5:
            d = Whitebear.get_differential(self, i)
        else:
            d = self.mu_disp[i-6, :]
        return d


if __name__ == "__main__":
    # Model testing

    pcs = get_thermopack_model("PC-SAFT")
    pcs.init("C1")
    PCS_functional = pc_saft(1, pcs, T_red=110.0/165.0)
    print(PCS_functional.d_hs[0], PCS_functional.T)
    dens_pcs = weighted_densities_pc_saft_1D(1, PCS_functional.R, ms=[1.0])

    v = pcs.specific_volume(PCS_functional.T,
                            1.0e6,
                            np.array([1.0]),
                            pcs.LIQPH)
    rho = (NA * PCS_functional.d_hs[0] ** 3)/v
    PCS_functional.test_bulk_differentials(rho)
    dens = weighted_densities_pc_saft_1D(1, PCS_functional.R, ms=[1.0])
    dens.set_testing_values(rho)
    # dens.print(print_utilities=True)
    PCS_functional.test_differentials(dens)
    corr = PCS_functional.get_bulk_correlation(rho)
    mu = PCS_functional.bulk_excess_chemical_potential(rho)
    print("corr, mu", corr, mu)

    # Hard sphere functionals
    # dens = weighted_densities_1D(1, 0.5)
    # dens.set_testing_values()
    # dens.print(print_utilities=True)
    #
    # RF_functional = Rosenfeld(N=1)
    # corr = RF_functional.get_bulk_correlation(np.array([rho]))
    # mu = RF_functional.bulk_excess_chemical_potential(np.array([rho]))
    # print("corr, mu", corr, mu)

    # RF_functional.test_differentials(dens)
    # WB_functional = Whitebear(N=1)
    # WB_functional.test_differentials(dens)
    # WBII_functional = WhitebearMarkII(N=1)
    # WBII_functional.test_differentials(dens)

    # rho = np.array([0.5, 0.1])
    # R = np.array([0.5, 0.3])
    # RF_functional = Rosenfeld(N=1, R=R)
    # RF_functional.test_bulk_differentials(rho)
    # WB_functional = Whitebear(N=1, R=R)
    # WB_functional.test_bulk_differentials(rho)
    # WBII_functional = WhitebearMarkII(N=1, R=R)
    # WBII_functional.test_bulk_differentials(rho)
