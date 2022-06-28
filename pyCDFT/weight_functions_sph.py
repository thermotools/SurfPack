#!/usr/bin/env python3
# This file is a proto-typing environment for spherical
import numpy as np
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utility import weighted_densities_1D, differentials_1D, \
    weighted_densities_pc_saft_1D, differentials_pc_saft_1D
import matplotlib.pyplot as plt
import pyfftw as fftw
from scipy.fft import dct, idct, dst, idst, fft, ifft
from scipy.special import spherical_jn
from weight_functions import Weights

class spherical_weights(Weights):
    """
    """

    def __init__(self, dr: float, R: float, N: int):
        """

        Args:
            dr (float): Grid spacing
            R (float): Particle radius
            N (int): Grid size
        """
        Weights.__init__(self, dr, R, N)

        # Preallocate grids
        self.z = np.zeros(N)
        self.k_cos = np.zeros(N)
        self.k_cos_R = np.zeros(N)
        self.k_sin = np.zeros(N)
        self.k_sin_R = np.zeros(N)

        # Fourier space variables
        self.fw3 = np.zeros(N, dtype=np.cdouble)
        self.fw2 = np.zeros(N, dtype=np.cdouble)
        self.fw2vec = np.zeros(N, dtype=np.cdouble)
        self.frho = np.zeros(N, dtype=np.cdouble)
        self.frho_delta = np.zeros(N, dtype=np.cdouble)
        self.frho_delta_cs = np.zeros(N)
        self.frho_delta_sph = np.zeros(N)
        self.frho_delta_cs_V = np.zeros(N)
        self.fw3_cs = np.zeros(N)
        self.fw2_cs = np.zeros(N)
        self.fw2vec_cs = np.zeros(N)
        self.fw3_sph = np.zeros(N)
        self.fw2_sph = np.zeros(N)
        self.fw2vec_sph = np.zeros(N)

        # Real space variables used for convolution
        self.rho = None
        self.rho_inf = np.zeros(1)
        self.d3_inf = np.zeros(1)
        self.d2eff_inf = np.zeros(1)
        self.d2veff_inf = np.zeros(1)

        self.rho_delta = np.zeros(N)
        self.d3_delta= np.zeros(N)
        self.d2eff_delta= np.zeros(N)
        self.d2veff_delta= np.zeros(N)

        # Extract analytics fourier transform of weights functions
        self.analytical_fourier_weigths()
        self.analytical_fourier_weigths_cs()
        self.analytical_fourier_weigths_sph()

        # The convolution of the fourier weights
        self.w2_conv=4.0*np.pi*(self.R**2)
        self.w3_conv=(4.0/3.0)*np.pi*(self.R**3)
        self.w2vec_conv=0.0
        self.w_disp_conv=1.0

        # Intermediate vectors for spherical convolutions
        self.n_V2_sph_part1=np.zeros(N)
        self.n_V2_sph_part2=np.zeros(N)
        self.n_V2_sph_prod2=np.zeros(N)
        self.n_V2_sph_prod2_V=np.zeros(N)

        # Intermediate variables for spherical corr. convolution
        self.fd2veff_sph_term1=np.zeros(N)
        self.fd2veff_sph_int_term2=np.zeros(N)
        self.fd2veff_sph_int_term2_V=np.zeros(N)

    def convolutions(self, densities: weighted_densities_1D, rho: np.ndarray):
        """
        We here calculate the convolutions for the weighted densities for the
        spherical geometry

        Args:
            densities:
            rho (np.ndarray): Density profile

        """

        # Split into two terms such that rho_delta=0 when z-> inf.
        self.rho_inf=rho[-1]
        self.rho_delta=rho-self.rho_inf

        # Fourier transfor only the rho_delta * r (the other term is done analytically)
        self.frho_delta_sph[:] = dst(self.rho_delta*self.z, type=2)

        # 2d weighted density (Spherical geometry)
        densities.fn2_delta_sph[:] = self.frho_delta_sph[:] * self.fw2_sph[:]
        densities.n2[:] = self.div_z*idst(densities.fn2_delta_sph, type=2)+self.rho_inf*self.w2_conv

        # 3d weighted density (Spherical geometry)
        densities.fn3_delta_sph[:] = self.frho_delta_sph[:] * self.fw3_sph[:]
        densities.n3[:] = self.div_z*idst(densities.fn3_delta_sph, type=2)+self.rho_inf*self.w3_conv

        # Vector 2d weighted density (Spherical geometry)
        self.n_V2_sph_part1=(self.div_z**2)*idst(self.frho_delta_sph[:] * self.fw3_sph[:], type=2)

        # Intermediate calculations for the intermediate term (Spherical geometry)
        self.n_V2_sph_prod2=self.fw2vec_sph*self.frho_delta_sph
        self.n_V2_sph_prod2_V = np.roll(self.n_V2_sph_prod2.copy(), 1)
        self.n_V2_sph_prod2_V[0] = 0
        self.n_V2_sph_part2=self.div_z*idct(self.n_V2_sph_prod2_V, type=2)

        densities.n2v[:] =self.n_V2_sph_part1-self.n_V2_sph_part2

        # Calculate remainig weighted densities after convolutions
        densities.update_after_convolution()

    def convolution_n3(self, densities: weighted_densities_1D, rho: np.ndarray):
        """

            Args:
                densities:
                rho (np.ndarray): Density profile

            Returns:

            """

        # Split into two terms such that rho_delta=0 when z-> inf.
        self.rho_inf=rho[-1]
        self.rho_delta=rho-self.rho_inf

        # Fourier transfor only the rho_delta * r (the other term is done analytically)
        self.frho_delta_sph[:] = dst(self.rho_delta*self.z, type=2)

        # 3d weighted density (Spherical geometry)
        densities.fn3_delta_sph[:] = self.frho_delta_sph[:] * self.fw3_sph[:]
        densities.n3[:] = self.div_z*idst(densities.fn3_delta_sph, type=2)+self.rho_inf*self.w3_conv

    def correlation_convolution(self, diff: differentials_1D):
        """

        Args:
            diff: Functional differentials

        Returns:

        """

        # Split all terms into (a_delta+a_inf) such that a_delta=0 when z-> inf.
        self.d3_inf=diff.d3[-1]
        self.d2eff_inf=diff.d2eff[-1]
        self.d2veff_inf=diff.d2veff[-1]

        self.d3_delta=diff.d3-self.d3_inf
        self.d2eff_delta=diff.d2eff-self.d2eff_inf
        self.d2veff_delta=diff.d2veff-self.d2veff_inf

        # First the scalar weights (spherical coordinates)
        diff.fd3_sph[:] = dst(self.d3_delta*self.z, type=2)
        diff.fd2eff_sph[:] = dst(self.d2eff_delta*self.z, type=2)

        # Fourier space multiplications (scalar terms)
        diff.fd3_conv_sph[:] = diff.fd3_sph[:]*self.fw3_sph[:]
        diff.fd2eff_conv_sph[:] = diff.fd2eff_sph[:]*self.fw2_sph[:]

        # Transform from Fourier space to real space
        diff.d3_conv[:] = self.div_z*idst(diff.fd3_conv_sph, type=2)+self.d3_inf*self.w3_conv
        diff.d2eff_conv[:] = self.div_z*idst(diff.fd2eff_conv_sph, type=2)+self.d2eff_inf*self.w2_conv

        # Next handle the vector term in spherical coordinates (has two terms)
        self.fd2veff_sph_term1=-1.0*dst(self.d2veff_delta, type=2)/(self.k_sin)
        self.fd2veff_sph_int_term2=dct(self.z*self.d2veff_delta, type=2)
        self.fd2veff_sph_int_term2_V=np.roll(self.fd2veff_sph_int_term2.copy(), -1)
        self.fd2veff_sph_int_term2_V[-1] = 0.0

        diff.fd2veff_conv_sph[:] = -1.0*(self.fd2veff_sph_term1[:]+self.fd2veff_sph_int_term2_V[:])*self.fw2vec_sph[:]
        diff.d2veff_conv[:] = self.div_z*idst(diff.fd2veff_conv_sph, type=2)

        diff.update_after_convolution()

    def analytical_fourier_weigths(self):
        """
        Analytical Fourier transform of w_2, w_3 and w_V2

        """
        # Fourier space variables
        kz = np.zeros(self.N)
        for k in range(int(self.N/2)):
            kz[k] = k
            kz[self.N - k - 1] = -k - 1

        kz /= self.dr*self.N
        kz_abs = np.zeros_like(kz)
        kz_abs[:] = np.abs(kz[:])
        kz_abs *= 2 * np.pi * self.R
        self.fw3.real[:] = (4.0/3.0) * np.pi * self.R**3 * \
            (spherical_jn(0, kz_abs) + spherical_jn(2, kz_abs))
        self.fw3.imag[:] = 0.0
        self.fw2.real[:] = 4 * np.pi * self.R**2 * spherical_jn(0, kz_abs)
        self.fw2.imag[:] = 0.0
        self.fw2vec.real[:] = 0.0
        self.fw2vec.imag[:] = -2 * np.pi * kz * self.fw3.real[:]

    def analytical_fourier_weigths_cs(self):
        """
        Analytical Fourier transform of w_2, w_3 and w_V2
        For the combined sine (imaginary) and cosine (real) transforms

        """

        self.z = np.linspace(self.dr/2, self.L - self.dr/2, self.N)
        self.k_cos = 2 * np.pi * np.linspace(0.0, self.N - 1, self.N) / (2 * self.L)
        self.k_sin = 2 * np.pi * np.linspace(1.0, self.N, self.N) / (2 * self.L)
        self.k_cos_R = self.k_cos*self.R
        self.k_sin_R = self.k_sin*self.R

        self.fw3_cs = 4/3 * np.pi * self.R**3 * \
            (spherical_jn(0, self.k_cos_R) + spherical_jn(2, self.k_cos_R))
        self.fw2_cs[:] = 4.0 * np.pi * self.R**2 * spherical_jn(0, self.k_cos_R)
        self.fw2vec_cs[:] = self.k_sin * \
            (4.0/3.0 * np.pi * self.R**3 * (spherical_jn(0, self.k_sin_R) + spherical_jn(2, self.k_sin_R)))

    def analytical_fourier_weigths_sph(self):
        """
        Analytical Fourier transform of w_2, w_3 and w_V2
        For the combined sine (imaginary) and cosine (real) transforms
        for a spherical geometry
        """
        self.z = np.linspace(self.dr/2, self.L - self.dr/2, self.N)
        self.div_z = (1/self.z)
        self.k_sin = 2 * np.pi * np.linspace(1.0, self.N, self.N) / (2 * self.L)
        self.k_sin_R = self.k_sin*self.R

        self.fw3_sph = 4/3 * np.pi * self.R**3 * \
            (spherical_jn(0, self.k_sin_R) + spherical_jn(2, self.k_sin_R))
        self.fw2_sph[:] = 4.0 * np.pi * self.R**2 * spherical_jn(0, self.k_sin_R)
        self.fw2vec_sph[:] = self.k_sin * self.fw3_sph


class spherical_pc_saft_weights(spherical_weights):
    """
    """

    def __init__(self, dr: float, R: float, N: int, phi_disp=1.3862):
        """

        Args:
            dr (float): Grid spacing
            R (float): Particle radius
            N (int): Grid size
            phi_disp (float): Weigthing distance for disperesion term
        """
        # Fourier space variables
        self.fw_disp = np.zeros(N, dtype=np.cdouble)
        self.frho_disp_delta = np.zeros(N, dtype=np.cdouble)
        self.fw_rho_disp_delta = np.zeros(N, dtype=np.cdouble)
        self.fmu_disp_delta = np.zeros(N, dtype=np.cdouble)
        self.fw_mu_disp_delta = np.zeros(N, dtype=np.cdouble)

        self.fw_disp_cs = np.zeros(N)
        self.frho_disp_delta_cs = np.zeros(N)
        self.fw_rho_disp_delta_cs = np.zeros(N)
        self.fmu_disp_delta_cs = np.zeros(N)
        self.fw_mu_disp_delta_cs = np.zeros(N)

        self.fw_disp_sph = np.zeros(N)
        self.frho_disp_delta_sph = np.zeros(N)
        self.fw_rho_disp_delta_sph = np.zeros(N)
        self.fmu_disp_delta_sph = np.zeros(N)
        self.fw_mu_disp_delta_sph = np.zeros(N)

        #  Regular arrays
        self.mu_disp_inf=np.zeros(N)
        self.mu_disp_delta=np.zeros(N)

        # Weigthing distance for disperesion term
        self.phi_disp = phi_disp

        spherical_weights.__init__(self, dr, R, N)

    def analytical_fourier_weigths(self):
        """
        Analytical Fourier transform of the dispersion weight

        """
        spherical_weights.analytical_fourier_weigths(self)
        # Fourier space variables
        kz = np.zeros(self.N)
        for k in range(int(self.N/2)):
            kz[k] = k
            kz[self.N - k - 1] = -k - 1

        kz /= self.dr*self.N
        kz_abs = np.zeros_like(kz)
        kz_abs[:] = np.abs(kz[:])
        kz_abs *= 4 * np.pi * self.R * self.phi_disp
        self.fw_disp.real = (spherical_jn(0, kz_abs) + spherical_jn(2, kz_abs))
        self.fw_disp.imag = 0.0

    def analytical_fourier_weigths_cs(self):
        """
        Analytical Fourier transform of w_2, w_3 and w_V2
        For the combined sine (imaginary) and cosine (real) transforms

        """

        # Extract anlytical transforms for fmt weight
        spherical_weights.analytical_fourier_weigths_cs(self)

        self.k_cos_R = 4 * np.pi *self.R*self.phi_disp*\
            np.linspace(0.0, self.N - 1, self.N) / (2 * self.L)

        self.fw_disp_cs = (spherical_jn(0, self.k_cos_R) + spherical_jn(2, self.k_cos_R))

    def analytical_fourier_weigths_sph(self):
        """
        Analytical Fourier transform of w_2, w_3 and w_V2
        For the 1D spherical transform

        """

        # Extract anlytical transforms for fmt weight
        planar_weights.analytical_fourier_weigths_sph(self)

        self.k_sin_R = 4 * np.pi *self.R*self.phi_disp*\
           np.linspace(1.0, self.N, self.N) / (2 * self.L)

        self.fw_disp_sph = (spherical_jn(0, self.k_sin_R) + spherical_jn(2, self.k_sin_R))

    def convolutions(self, densities: weighted_densities_pc_saft_1D, rho: np.ndarray):
        """

        Args:
            densities:
            rho (np.ndarray): Density profile

        """
        spherical_weights.convolutions(self, densities, rho)

        # Split into two terms such that rho_delta=0 when z-> inf.
        self.rho_inf=rho[-1]
        self.rho_delta=rho-self.rho_inf

        # Fourier transform only the rho_delta (the other term is done analytically)
        self.frho_disp_delta_sph[:] = dst(self.z*self.rho_delta, type=2)

         # Dispersion density
        self.fw_rho_disp_delta_sph[:] = self.frho_disp_delta_sph[:] * self.fw_disp_sph[:]
        densities.rho_disp[:] = self.div_z*idst(self.fw_rho_disp_delta_sph, type=2)+self.rho_inf*self.w_disp_conv


    def correlation_convolution(self, diff: differentials_pc_saft_1D):
        """

        Args:
            diff: Functional differentials

        Returns:

        """
        spherical_weights.correlation_convolution(self, diff)

        # Split the term into (a_delta+a_inf) such that a_delta=0 when z-> inf.
        self.mu_disp_delta=diff.mu_disp-self.mu_disp_inf

        # Fourier transform derivatives
        self.fmu_disp_delta_sph[:] = dst(self.mu_disp_delta, type=2)

        # Fourier space multiplications
        self.fw_mu_disp_delta_sph[:] = self.fmu_disp_delta_sph[:] * self.fw_disp_sph[:]

        # Transform from Fourier space to real space
        diff.mu_disp_conv[:] = self.div_z*idst(self.fw_mu_disp_delta_sph, type=2)+\
            self.mu_disp_inf*self.w_disp_conv

        diff.update_after_convolution()

if __name__ == "__main__":
    pass
