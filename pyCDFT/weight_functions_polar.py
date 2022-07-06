#!/usr/bin/env python3
# This file is a proto-typing environment for spherical
import numpy as np
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from utility import weighted_densities_1D, differentials_1D, \
    weighted_densities_pc_saft_1D, differentials_pc_saft_1D
import matplotlib.pyplot as plt
import pyfftw as fftw
from scipy.fft import dct, idct, dst, idst, fft, ifft
from scipy.special import jv
from weight_functions import Weights

def polar(object):

    def __init__(self,
                 domain_size=15.0,
                 n_grid=1024):
        """Set up grid according tO Xi eta al. 2020
        An Efficient Algorithm for Molecular Density Functional Theory in Cylindrical Geometry:
        Application to Interfacial Statistical Associating Fluid Theory (iSAFT)
        DOI: 10.1021/acs.iecr.9b06895
        """
        alpha = 0.002
        for _ in range(21):
            alpha = -np.log(1.0 - np.exp(-alpha)) / (n_grid - 1)
        self.alpha = alpha
        self.x0 = 0.5 * (np.exp(-alpha * n_grid) + np.exp(-alpha * (n_grid - 1)))
        # Setting the grid
        self.z = np.zeros(n_grid)
        for i in range(n_grid):
            self.z = domain_size*self.x0*np.exp(alpha*i)
        # Setting the edge grid
        self.z_edge = np.zeros(n_grid+1)
        for i in range(1,n_grid+1):
            self.z_edge = domain_size*np.exp(-alpha*(n_grid-i))
        # End correction factor
        k0 = np.exp(2*alpha)*(2*np.exp(alpha) + np.exp(2*alpha) - 1)/ \
            (1 + np.exp(alpha))**2/(np.exp(2*alpha) - 1)
        k0v = np.exp(2*alpha)*(2*np.exp(alpha) + np.exp(2*alpha) - 5.0/3.0)/ \
            (1 + np.exp(alpha))**2/(np.exp(2*alpha) - 1)
        self.k = np.ones(n_grid)
        self.k[0] = k0
        self.kv = np.ones(n_grid)
        self.kv[0] = k0v
        # Hankel paramaters
        self.b = domain_size
        #fac = 1.0/(2*self.x0*(np.exp(alpha*(n_grid-1)) - np.exp(alpha*(n_grid-2))))
        #self.lam = int(0.5*fac/self.b)
        #self.gamma = self.lam*self.b
        # Defining integration weights
        self.integration_weights = np.zeros(n_grid)
        self.integration_weights[0] = k0 * np.exp(2 * alpha)
        self.integration_weights[1] = (np.exp(2 * alpha) - k0) * np.exp(2 * alpha)
        for i in range(2,n_grid):
            self.integration_weights[i] = np.exp(2 * alpha * i) * (np.exp(2 * alpha) - 1.0)
        self.integration_weights *= np.exp(-2 * alpha * n_grid) * np.pi * domain_size**2

class polar_weights(Weights):
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
        self.pol = polar(domain_size=self.L,n_grid=N)

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

    def fourier_weigths(self):
        """
        Fourier transform of w_2, w_3 and w_V2

        """
        n_grid = self.n_grid
        x0 = self.pol.x0
        alpha = self.alpha
        gamma = np.exp(alpha * (n_grid - 1))
        l = self.L
        k_grid = np.zeros(n_grid)
        for i in range(n_grid):
            k_grid[i] = x0 * np.exp(alpha * i) * gamma / l

        jv1 = np.zeros(2*n_grid, dtype=np.cdouble)
        jv2 = np.zeros(2*n_grid, dtype=np.cdouble)
        for i in range(2*n_grid):
            j[i] = jv(1, gamma * x0 * (alpha * np.exp((i + 1) - n_grid))) / (2 * n_grid)
            jv[i] = jv(2, gamma * x0 * (alpha * np.exp((i + 1) - n_grid))) / (2 * n_grid)

        self.jv1 = sfft.ifft(jv1)
        self.jv2 = sfft.ifft(jv2)


class polar_pc_saft_weights(polar_weights):
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

        polar_weights.__init__(self, dr, R, N)

    def analytical_fourier_weigths(self):
        """
        Analytical Fourier transform of the dispersion weight

        """
        polar_weights.analytical_fourier_weigths(self)
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

    def convolutions(self, densities: weighted_densities_pc_saft_1D, rho: np.ndarray):
        """

        Args:
            densities:
            rho (np.ndarray): Density profile

        """
        polar_weights.convolutions(self, densities, rho)

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
        polar_weights.correlation_convolution(self, diff)

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
