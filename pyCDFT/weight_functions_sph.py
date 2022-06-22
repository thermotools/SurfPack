#!/usr/bin/env python3
# This file is a proto-typing environment for spherical
import numpy as np
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from scipy.ndimage import convolve1d
from utility import weighted_densities_1D, differentials_1D, \
    weighted_densities_pc_saft_1D, differentials_pc_saft_1D
import matplotlib.pyplot as plt
import pyfftw as fftw
from scipy.fft import dct, idct, dst, idst, fft, ifft
from scipy.special import spherical_jn

class planar_weights():
    """
    """

    def __init__(self, dr: float, R: float, N: int, quad="Roth"):
        """

        Args:
            dr (float): Grid spacing
            R (float): Particle radius
            N (int): Grid size
            quad (str): Quadrature for integral
        """
        self.dr = dr
        self.R = R
        self.N = N
        self.L = self.dr*self.N    # Length

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

        # Fourier transfor only the rho_delta (the other term is done analytically)
        self.frho_delta_sph[:] = dct(self.rho_delta, type=2)

        # 2d weighted density (Cosine transformSpheri)
        densities.fn2_delta_cs[:] = self.frho_delta_cs[:] * self.fw2_cs[:]
        densities.n2[:] = idct(densities.fn2_delta_cs, type=2)+self.rho_inf*self.w2_conv

        
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
        diff.fd2veff_cs[:] = dct(self.d2veff_delta, type=2)

        # Shift the fourier transform for the sine-transform
        diff.fd2veff_cs_V=np.roll(diff.fd2veff_cs.copy(), -1)
        diff.fd2veff_cs_V[-1] = 0
       
        # Fourier space multiplications
        diff.fd3_conv_cs[:] = diff.fd3_cs[:] * self.fw3_cs[:]
        diff.fd2eff_conv_cs[:] = diff.fd2eff_cs[:] * self.fw2_cs[:]
        diff.fd2veff_conv_cs[:] = diff.fd2veff_cs_V[:] * (-1.0 * self.fw2vec_cs[:])
        
        # Transform from Fourier space to real space
        diff.d3_conv[:] = idct(diff.fd3_conv_cs, type=2)+self.d3_inf*self.w3_conv
        diff.d2eff_conv[:] = idct(diff.fd2eff_conv_cs, type=2)+self.d2eff_inf*self.w2_conv
        diff.d2veff_conv[:] = idst(diff.fd2veff_conv_cs, type=2)

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
        self.k_sin = 2 * np.pi * np.linspace(1.0, self.N, self.N) / (2 * self.L)
        self.k_sin_R = self.k_sin*self.R

        self.fw3_sph = 4/3 * np.pi * self.R**3 * \
            (spherical_jn(0, self.sin_R) + spherical_jn(2, self.k_sin_R))
        self.fw2_sph[:] = 4.0 * np.pi * self.R**2 * spherical_jn(0, self.k_sin_R)
        self.fw2vec_sph[:] = self.k_sin * \
            (4.0/3.0 * np.pi * self.R**3 * (spherical_jn(0, self.k_sin_R) + spherical_jn(2, self.k_sin_R)))
        
class planar_pc_saft_weights(planar_weights):
    """
    """

    def __init__(self, dr: float, R: float, N: int, quad="Roth", phi_disp=1.3862):
        """

        Args:
            dr (float): Grid spacing
            R (float): Particle radius
            N (int): Grid size
            quad (str): Quadrature for integral
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

        planar_weights.__init__(self, dr, R, N, quad)

    def analytical_fourier_weigths(self):
        """
        Analytical Fourier transform of the dispersion weight

        """
        planar_weights.analytical_fourier_weigths(self)
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
        planar_weights.analytical_fourier_weigths_cs(self)
        
        self.k_cos_R = 4 * np.pi *self.R*self.phi_disp*\
            np.linspace(0.0, self.N - 1, self.N) / (2 * self.L)

        self.fw_disp_cs = (spherical_jn(0, self.k_cos_R) + spherical_jn(2, self.k_cos_R))

    def convolutions(self, densities: weighted_densities_pc_saft_1D, rho: np.ndarray):
        """

        Args:
            densities:
            rho (np.ndarray): Density profile

        """
        planar_weights.convolutions(self, densities, rho)

        # Split into two terms such that rho_delta=0 when z-> inf.
        self.rho_inf=rho[-1]
        self.rho_delta=rho-self.rho_inf

        # Fourier transform only the rho_delta (the other term is done analytically)
        self.frho_disp_delta_cs[:] = dct(self.rho_delta, type=2)
        
         # Dispersion density
        self.fw_rho_disp_delta_cs[:] = self.frho_disp_delta_cs[:] * self.fw_disp_cs[:]
        densities.rho_disp[:] = idct(self.fw_rho_disp_delta_cs, type=2)+self.rho_inf*self.w_disp_conv
        
    def correlation_convolution(self, diff: differentials_pc_saft_1D):
        """

        Args:
            diff: Functional differentials

        Returns:

        """
        planar_weights.correlation_convolution(self, diff)

        # Split the term into (a_delta+a_inf) such that a_delta=0 when z-> inf.
        self.mu_disp_delta=diff.mu_disp-self.mu_disp_inf

        # Fourier transform derivatives
        self.fmu_disp_delta_cs[:] = dct(self.mu_disp_delta, type=2)

        # Fourier space multiplications
        self.fw_mu_disp_delta_cs[:] = self.fmu_disp_delta_cs[:] * self.fw_disp_cs[:]

        # Transform from Fourier space to real space
        diff.mu_disp_conv[:] = idct(self.fw_mu_disp_delta_cs, type=2)+\
            self.mu_disp_inf*self.w_disp_conv
        
        diff.update_after_convolution()


class planar_weights_system_mc():
    """
    Multicomponent planar weigths
    """

    def __init__(self, functional,
                 dr: float,
                 R: np.ndarray,
                 N: int,
                 quad="Roth",
                 mask_conv_results=None,
                 ms=None,
                 plweights=planar_weights,
                 wd=weighted_densities_1D,
                 diff=differentials_1D):
        """

        Args:
            functional: Functional
            dr (float): Grid spacing
            R (ndarray): Particle radius
            N (int): Grid size
            quad (str): Quadrature for integral
            mask_conv_results: Mask to avoid divide by zero
            ms (np.ndarray): Mumber of monomers
            plweights: Class type,defaults to planar_weights
            wd: Class type,defaults to weighted_densities_1D
            diff: Class type,defaults to differentials_1D
        """
        self.functional = functional
        self.pl_weights = []
        self.comp_weighted_densities = []
        self.comp_differentials = []
        self.nc = np.shape(R)[0]
        if ms is None:
            ms = np.ones(self.nc)
        for i in range(self.nc):
            self.pl_weights.append(plweights(dr=dr, R=R[i], N=N, quad=quad))
            self.comp_weighted_densities.append(
                wd(N=N, R=R[i], ms=ms[i], mask_conv_results=mask_conv_results))
            self.comp_differentials.append(diff(N=N, R=R[i]))

        # Overall weighted densities
        self.weighted_densities = wd(N=N, R=0.5, ms=ms[0])  # Dummy R

    def convolutions(self, rho):
        """
        Perform convolutions for weighted densities

        Args:
            rho (array_like): Density profile
        """
        self.weighted_densities.set_zero()
        for i in range(self.nc):
            self.pl_weights[i].convolutions(
                self.comp_weighted_densities[i], rho[i])
            self.weighted_densities += self.comp_weighted_densities[i]
        self.weighted_densities.update_utility_variables()

    def convolution_n3(self, rho):
        """

            Args:
                rho (array_like): Density profile

            Returns:

            """
        self.weighted_densities.set_zero()
        for i in range(self.nc):
            self.pl_weights[i].convolution_n3(
                self.comp_weighted_densities[i], rho[i])
            self.weighted_densities += self.comp_weighted_densities[i]
        # self.weighted_densities.update_after_convolution()

    def correlation_convolution(self):
        """
        Calculate functional differentials and perform convolutions with the
        appropriate weight functions.
        """
        self.functional.differentials(self.weighted_densities)
        for i in range(self.nc):
            self.comp_differentials[i].set_functional_differentials(
                self.functional, i)
            self.pl_weights[i].correlation_convolution(
                self.comp_differentials[i])


class planar_weights_system_mc_pc_saft(planar_weights_system_mc):
    """
    Multicomponent planar weigths including PC-SAFT dispersion
    """

    def __init__(self, functional,
                 dr: float,
                 R: np.ndarray,
                 N: int,
                 pcsaft: object,
                 mask_conv_results=None,
                 plweights=planar_pc_saft_weights,
                 wd=weighted_densities_pc_saft_1D,
                 diff=differentials_pc_saft_1D):
        """

        Args:
            functional: Functional
            dr (float): Grid spacing
            R (np.ndarray): Particle radius
            N (int): Grid size
            pcsaft (pyctp.pcsaft): Thermopack object
            mask_conv_results: Mask to avoid divide by zero
            plweights: Class type,defaults to planar_pc_saft_weights
            wd: Class type,defaults to weighted_densities_pc_saft_1D
            diff: Class type,defaults to differentials_pc_saft_1D
        """

        self.thermo = pcsaft
        planar_weights_system_mc.__init__(self,
                                          functional,
                                          dr,
                                          R,
                                          N,
                                          quad="None",
                                          mask_conv_results=mask_conv_results,
                                          ms=pcsaft.m,
                                          plweights=plweights,
                                          wd=wd,
                                          diff=diff)
        # Make sure rho_disp_array is allocated for mother density class
        self.weighted_densities.rho_disp_array = np.zeros((self.nc, N))

    def convolutions(self, rho):
        """
        Perform convolutions for weighted densities

        Args:
            rho (array_like): Density profile
        """
        planar_weights_system_mc.convolutions(self, rho)
        for i in range(self.nc):
            self.weighted_densities.rho_disp_array[i, :] = \
                self.comp_weighted_densities[i].rho_disp[:]


if __name__ == "__main__":
    plw = planar_weights(dr=0.25, R=0.5, N=8, quad="None")
    plw.plot_weights()
    plw.plot_actual_weights()
