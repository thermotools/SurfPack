#!/usr/bin/env python3
import numpy as np
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utility import weighted_densities_1D, differentials_1D, \
    weighted_densities_pc_saft_1D, differentials_pc_saft_1D
import matplotlib.pyplot as plt
import pyfftw as fftw
from scipy.fft import dct, idct, dst, idst, fft, ifft
from scipy.special import spherical_jn
from weight_functions import Weights

class planar_cosine_sine_weights(Weights):
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
        self.frho_delta_cs_V = np.zeros(N)
        self.fw3_cs = np.zeros(N)
        self.fw2_cs = np.zeros(N)
        self.fw2vec_cs = np.zeros(N)

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

        # The convolution of the fourier weights
        self.w2_conv=4.0*np.pi*(self.R**2)
        self.w3_conv=(4.0/3.0)*np.pi*(self.R**3)
        self.w2vec_conv=0.0
        self.w_disp_conv=1.0

    def convolutions(self, densities: weighted_densities_1D, rho: np.ndarray):
        """

        Args:
            densities:
            rho (np.ndarray): Density profile

        """

         # Split into two terms such that rho_delta=0 when z-> inf.
        self.rho_inf=rho[-1]
        self.rho_delta=rho-self.rho_inf

        # Fourier transfor only the rho_delta (the other term is done analytically)
        self.frho_delta_cs[:] = dct(self.rho_delta, type=2)
        self.frho_delta_cs_V = np.roll(self.frho_delta_cs.copy(), -1) # Fourier transform of density profile for `k_sin` 
        self.frho_delta_cs_V[-1] = 0                                  # this information is lost but usually not important

        # 2d weighted density (Cosine transform)
        densities.fn2_delta_cs[:] = self.frho_delta_cs[:] * self.fw2_cs[:]
        densities.n2[:] = idct(densities.fn2_delta_cs, type=2)+self.rho_inf*self.w2_conv

        # 3d weighted density (Cosine transform)
        densities.fn3_delta_cs[:] = self.frho_delta_cs[:] * self.fw3_cs[:]
        densities.n3[:] = idct(densities.fn3_delta_cs, type=2)+self.rho_inf*self.w3_conv

        # Vector 2d weighted density (Sine transform)
        densities.fn2v_delta_cs[:] = self.frho_delta_cs_V[:] * self.fw2vec_cs[:]
        densities.n2v[:] = idst(densities.fn2v_delta_cs, type=2)

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

        # Fourier transfor only the rho_delta (the other term is done analytically)
        self.frho_delta_cs[:] = dct(self.rho_delta, type=2)

        # 3d weighted density (Cosine transform)
        densities.fn3_delta_cs[:] = self.frho_delta_cs[:] * self.fw3_cs[:]
        densities.n3[:] = idct(densities.fn3_delta_cs, type=2)+self.rho_inf*self.w3_conv


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
   
        # Fourier transform of delta derivatives (sine and cosine)
        diff.fd3_cs[:] = dct(self.d3_delta, type=2)
        diff.fd2eff_cs[:] = dct(self.d2eff_delta, type=2)
        diff.fd2veff_cs[:] = dst(self.d2veff_delta, type=2)       # The vector valued function is odd

        # Fourier space multiplications
        diff.fd3_conv_cs[:] = diff.fd3_cs[:] * self.fw3_cs[:]
        diff.fd2eff_conv_cs[:] = diff.fd2eff_cs[:] * self.fw2_cs[:]
        diff.fd2veff_conv_cs[:] = diff.fd2veff_cs[:] * self.fw2vec_cs[:]

        # We must roll the vector to conform with the cosine transform
        diff.fd2veff_conv_cs_V[:]=np.roll(diff.fd2veff_conv_cs.copy(),1)
        diff.fd2veff_conv_cs_V[0]=0
                
        # Transform from Fourier space to real space
        diff.d3_conv[:] = idct(diff.fd3_conv_cs, type=2)+self.d3_inf*self.w3_conv
        diff.d2eff_conv[:] = idct(diff.fd2eff_conv_cs, type=2)+self.d2eff_inf*self.w2_conv
        diff.d2veff_conv[:] = idct(diff.fd2veff_conv_cs_V, type=2)
 
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

class planar_cosine_sine_pc_saft_weights(planar_cosine_sine_weights):
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

        #  Regular arrays
        self.mu_disp_inf=np.zeros(N)
        self.mu_disp_delta=np.zeros(N)

        # Weigthing distance for disperesion term
        self.phi_disp = phi_disp

        planar_cosine_sine_weights.__init__(self, dr, R, N)

    def analytical_fourier_weigths(self):
        """
        Analytical Fourier transform of the dispersion weight

        """
        planar_cosine_sine_weights.analytical_fourier_weigths(self)
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
        planar_cosine_sine_weights.analytical_fourier_weigths_cs(self)

        self.k_cos_R = 4 * np.pi *self.R*self.phi_disp*\
            np.linspace(0.0, self.N - 1, self.N) / (2 * self.L)

        self.fw_disp_cs = (spherical_jn(0, self.k_cos_R) + spherical_jn(2, self.k_cos_R))

    def convolutions(self, densities: weighted_densities_pc_saft_1D, rho: np.ndarray):
        """

        Args:
            densities:
            rho (np.ndarray): Density profile

        """
        planar_cosine_sine_weights.convolutions(self, densities, rho)

        # Split into two terms such that rho_delta=0 when z-> inf.
        self.rho_inf=rho[-1]
        self.rho_delta[:] = rho - self.rho_inf
        #print("self.rho_delta[:]",self.rho_delta[:])
        #print("self.rho_inf", self.rho_inf)
        # Fourier transform only the rho_delta (the other term is done analytically)
        self.frho_disp_delta_cs[:] = dct(self.rho_delta, type=2)

         # Dispersion density
        self.fw_rho_disp_delta_cs[:] = self.frho_disp_delta_cs[:] * self.fw_disp_cs[:]
        densities.rho_disp[:] = idct(self.fw_rho_disp_delta_cs, type=2)+self.rho_inf*self.w_disp_conv
        #print(densities.rho_disp)

    def correlation_convolution(self, diff: differentials_pc_saft_1D):
        """

        Args:
            diff: Functional differentials

        Returns:

        """
        planar_cosine_sine_weights.correlation_convolution(self, diff)

        # Split the term into (a_delta+a_inf) such that a_delta=0 when z-> inf.
        self.mu_disp_inf = diff.mu_disp[-1]
        self.mu_disp_delta=diff.mu_disp-self.mu_disp_inf

        # Fourier transform derivatives
        self.fmu_disp_delta_cs[:] = dct(self.mu_disp_delta, type=2)

        # Fourier space multiplications
        self.fw_mu_disp_delta_cs[:] = self.fmu_disp_delta_cs[:] * self.fw_disp_cs[:]

        # Transform from Fourier space to real space
        diff.mu_disp_conv[:] = idct(self.fw_mu_disp_delta_cs, type=2)+\
            self.mu_disp_inf*self.w_disp_conv
        #print("self.mu_disp_inf, self.w_disp_conv", self.mu_disp_inf,self.w_disp_conv)
        diff.update_after_convolution()

if __name__ == "__main__":
    pass
