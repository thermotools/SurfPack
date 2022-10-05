#!/usr/bin/env python3
import numpy as np
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from constants import NA, RGAS, Geometry, DftEnum
from scipy.fft import dct, idct, dst, idst, fft, ifft
from scipy.special import spherical_jn
from sympy import sympify, lambdify

class WeightFunctionType(DftEnum):
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
        wf_type (WeightFunctionType): Type of weight
        kernel_radius (float): Relative width of weight
        alias (string): Alias for acessing weight
        prefactor (string): Weight prefactor as function of R and Psi (Psi=kernel_radius)
        convolve (bool): Perform convolution= Default: True
        calc_from (string): Calculate from another weight? Default: None
        """
        assert (convolve==True and calc_from is None) or (convolve==False and calc_from is not None)
        self.wf_type = wf_type
        self.kernel_radius = kernel_radius
        self.alias = alias
        self.prefactor_str = prefactor
        self.prefactor = lambdify(("R,Psi"), sympify(prefactor), "numpy")
        self.prefactor_R = lambdify(("R,Psi"), sympify(prefactor).diff("R"), "numpy")
        if wf_type == WeightFunctionType.DELTA:
            integral = "4.0*pi*R**2*Psi**2"
        elif wf_type == WeightFunctionType.THETA:
            integral = "4.0*pi*R**3*Psi**3/3.0"
        elif wf_type == WeightFunctionType.NORMTHETA:
            integral = "1.0"
        elif wf_type == WeightFunctionType.DELTAVEC:
            integral = "2*pi" # Dummy
        self.lamb = lambdify(("R,Psi"), sympify(integral+"*"+prefactor), "numpy")
        self.lamb_R = lambdify(("R,Psi"), sympify(integral+"*"+prefactor).diff("R"), "numpy")
        self.lamb_RR = lambdify(("R,Psi"), sympify(integral+"*"+prefactor).diff("R", 2), "numpy")
        self.convolve = convolve
        self.calc_from = calc_from
        # For transformations
        self.prefactor_evaluated = None
        self.R = None
        self.fw = None
        self.w_conv_steady = None
        self.k_grid = None
        self.k_cos = None
        self.k_cos_R = None
        self.k_sin = None
        self.k_sin_R = None
        self.one_div_r = None
        self.r = None
        self.geometry = None
        self.fn = None
        # Entropy weights
        self.fw_T = None
        self.w_conv_steady_T = None

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

    def generate_fourier_weights(self, grid, R, R_T):
        """
        """
        self.R = R
        self.R_T = R_T
        self.geometry = grid.geometry
        self.fn = np.zeros(grid.n_grid)
        self.prefactor_evaluated = self.prefactor(R,self.kernel_radius)
        if self.geometry == Geometry.PLANAR:
            self.generate_planar_fourier_weights(grid, R, R_T)
        elif self.geometry == Geometry.SPHERICAL:
            self.generate_spherical_fourier_weights(grid, R, R_T)
        elif self.geometry == Geometry.POLAR:
            self.generate_polar_fourier_weights(grid, R, R_T)

    def generate_planar_fourier_weights(self, grid, R, R_T):
        """
        """
        self.fw = np.zeros(grid.n_grid)
        self.k_cos = np.zeros(grid.n_grid)
        self.k_cos_R = np.zeros(grid.n_grid)
        self.k_sin = np.zeros(grid.n_grid)
        self.k_sin_R = np.zeros(grid.n_grid)
        N = grid.n_grid
        L = grid.domain_size
        R_kernel = R*self.kernel_radius
        self.k_cos = 2 * np.pi * np.linspace(0.0, N - 1, N) / (2 * L)
        self.k_cos_R[:] = self.k_cos * R_kernel
        self.k_sin = 2 * np.pi * np.linspace(1.0, N, N) / (2 * L)
        self.k_sin_R[:] = self.k_sin * R_kernel
        self.generate_planar_fourier_weights_T(grid, R, R_T)
        if self.convolve:
            if self.wf_type == WeightFunctionType.THETA:
                self.w_conv_steady = self.lamb(R,self.kernel_radius)
                self.fw[:] = self.w_conv_steady * \
                    (spherical_jn(0, self.k_cos_R) + spherical_jn(2, self.k_cos_R))
            elif self.wf_type == WeightFunctionType.NORMTHETA:
                self.w_conv_steady = 1.0
                self.fw[:] = spherical_jn(0, self.k_cos_R) + spherical_jn(2, self.k_cos_R)
            elif self.wf_type == WeightFunctionType.DELTA:
                # The convolution of the fourier weights for steady profile
                self.w_conv_steady = self.lamb(R,self.kernel_radius)
                self.fw[:] = self.w_conv_steady * spherical_jn(0, self.k_cos_R)
            elif self.wf_type == WeightFunctionType.DELTAVEC:
                # The convolution of the fourier weights for steady profile
                self.w_conv_steady = 0.0
                self.fw[:] = - self.prefactor_evaluated * self.k_sin * \
                    (4.0/3.0 * np.pi * R_kernel**3 * (spherical_jn(0, self.k_sin_R) + spherical_jn(2, self.k_sin_R)))
                # elif self.alias == "wv1":
                #     print("Setting vw1")
                #     self.fw[:] = - (1.0/(4*np.pi*R_kernel))*self.k_sin * \
                #         (4.0/3.0 * np.pi * R_kernel**3 * (spherical_jn(0, self.k_sin_R) + spherical_jn(2, self.k_sin_R)))

    def generate_planar_fourier_weights_T(self, grid, R, R_T):
        """
        """
        self.fw_T = np.zeros(grid.n_grid)
        R_kernel = R*self.kernel_radius
        if self.wf_type == WeightFunctionType.THETA:
            self.w_conv_steady_T = self.lamb_R(R,self.kernel_radius)
            self.fw_T[:] = self.w_conv_steady_T * spherical_jn(0, self.k_cos_R)
            beta_R = self.prefactor_R(R,self.kernel_radius)
            if beta_R != 0.0:
                self.fw_T[:] += beta_R*self.fw[:]/self.prefactor_evaluated
        elif self.wf_type == WeightFunctionType.NORMTHETA:
            self.fw_T[:] = -3*spherical_jn(2, self.k_cos_R)/R
            self.w_conv_steady_T = 0.0
        elif self.wf_type == WeightFunctionType.DELTA:
            lamb = self.lamb(R,self.kernel_radius)
            lamb_R = self.lamb_R(R,self.kernel_radius)
            self.fw_T[:] = lamb_R * spherical_jn(0, self.k_cos_R) - self.kernel_radius*lamb*self.k_cos*spherical_jn(1, self.k_cos_R)
            self.w_conv_steady_T = lamb_R
        elif self.wf_type == WeightFunctionType.DELTAVEC:
            if self.lamb_R(R,self.kernel_radius) == 0.0:
               self.fw_T[:] = - 4 * np.pi * self.k_sin_R * R_kernel * spherical_jn(0, self.k_sin_R)
            else:
                self.fw_T[:] = - self.k_sin_R * spherical_jn(0, self.k_sin_R) + spherical_jn(1, self.k_sin_R)
            self.w_conv_steady_T = 0.0
        self.fw_T[:] *= R_T
        self.w_conv_steady_T *= R_T

    def generate_spherical_fourier_weights(self, grid, R, R_T):
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
        if self.convolve:
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

    def generate_polar_fourier_weights(self, grid, R, R_T):
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

    def planar_convolution_T(self, rho_inf: float, frho_delta: np.ndarray, weighted_density: np.ndarray):
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
            self.fn[:] = frho_delta_V[:] * self.fw_T[:]
            weighted_density[:] = idst(self.fn, type=2)
        else:
            # Cosine transform
            self.fn[:] = frho_delta[:] * self.fw_T[:]
            weighted_density[:] = idct(self.fn, type=2) + rho_inf*self.w_conv_steady_T

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

    def planar_convolution_differentials_T(self, diff: np.ndarray, diff_conv: np.ndarray):
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
            self.fn[:] = fd_delta[:] * self.fw_T[:]
            # We must roll the vector to conform with the cosine transform
            self.fn = np.roll(self.fn, 1)
            self.fn[0] = 0
        else:
            fd_delta = dct(d_delta, type=2)
            self.fn[:] = fd_delta[:] * self.fw_T[:]

        diff_conv[:] = idct(self.fn, type=2) + d_inf*self.w_conv_steady_T


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

    def convolve_differentials_T(self, diff: np.ndarray, conv_diff: np.ndarray):
        """

        Args:
            densities:
            rho (np.ndarray): Density profile

        """
        if self.geometry == Geometry.PLANAR:
            self.planar_convolution_differentials_T(diff, conv_diff)
        else:
            pass

    def convolve_densities_T(self, rho_inf: float, frho_delta: np.ndarray, weighted_density: np.ndarray):
        """

        Args:
            densities:
            rho (np.ndarray): Density profile

        """
        if self.geometry == Geometry.PLANAR:
            self.planar_convolution_T(rho_inf, frho_delta, weighted_density)
        else:
            pass

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
        # self.wfs["wv1"] = WeightFunction(WeightFunctionType.DELTAVEC,
        #                                  kernel_radius=1.0,
        #                                  alias = "wv1",
        #                                  prefactor = "1.0",
        #                                  convolve=True,
        #                                  calc_from=None)
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

    def add_weight(self, alias, kernel_radius, wf_type, prefactor, convolve=True, calc_from=None):
        self.wfs[alias] = WeightFunction(wf_type=wf_type,
                                         kernel_radius=kernel_radius,
                                         alias=alias,
                                         prefactor=prefactor,
                                         convolve=convolve,
                                         calc_from=calc_from)

    def get_correlation_factor(self, label, R):
        """
        """
        corr_fac = 0.0
        for wf in self.wfs:
            if self.wfs[wf].convolve and wf == label:
                corr_fac += 1.0
            elif self.wfs[wf].calc_from == label:
                corr_fac += self.wfs[wf].prefactor(R,self.wfs[wf].kernel_radius)
        return corr_fac



if __name__ == "__main__":
    pass
