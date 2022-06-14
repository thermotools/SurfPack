#!/usr/bin/env python3
import numpy as np
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from scipy.ndimage import convolve1d
from utility import weighted_densities_1D, differentials_1D, \
    weighted_densities_pc_saft_1D, differentials_pc_saft_1D
import matplotlib.pyplot as plt
import pyfftw as fftw
import scipy.fft as sfft
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

        # Fourier space variables
        self.fw3 = np.zeros(N, dtype=np.cdouble)
        self.fw2 = np.zeros(N, dtype=np.cdouble)
        self.fw2vec = np.zeros(N, dtype=np.cdouble)
        self.frho = np.zeros(N, dtype=np.cdouble)

        # Real space variables used for convolution
        self.rho = None

        # Extract analytics fourier transform of weights functions
        self.analytical_fourier_weigths()

        # Fourier transformation objects. Allocated in separate method.
        self.fftw_rho = None
        self.ifftw_n2 = None
        self.ifftw_n3 = None
        self.ifftw_n2v = None
        self.fftw_d2eff = None
        self.fftw_d3 = None
        self.fftw_d2veff = None
        self.ifftw_d2eff = None
        self.ifftw_d3 = None
        self.ifftw_d2veff = None

    def convolutions(self, densities: weighted_densities_1D, rho: np.ndarray):
        """

        Args:
            densities:
            rho (np.ndarray): Density profile

        """
        self.frho[:] = sfft.fft(rho)
        # 2d weighted density
        densities.fn2[:] = self.frho[:] * self.fw2[:]
        densities.n2[:] = sfft.ifft(densities.fn2).real

        # 3d weighted density
        densities.fn3[:] = self.frho[:] * self.fw3[:]
        densities.n3[:] = sfft.ifft(densities.fn3).real

        # Vector 2d weighted density
        densities.fn2v[:] = self.frho[:] * self.fw2vec[:]
        densities.n2v[:] = sfft.ifft(densities.fn2v).real

        densities.update_after_convolution()

    def convolution_n3(self, densities: weighted_densities_1D, rho: np.ndarray):
        """

            Args:
                densities:
                rho (np.ndarray): Density profile

            Returns:

            """
        self.frho[:] = sfft.fft(rho)
        # 3d weighted density
        densities.fn3[:] = self.frho[:] * self.fw3[:]
        densities.n3[:] = sfft.ifft(densities.fn3).real

    def correlation_convolution(self, diff: differentials_1D):
        """

        Args:
            diff: Functional differentials

        Returns:

        """
        # Fourier transform derivatives
        diff.fd3[:] = sfft.fft(diff.d3)
        diff.fd2eff[:] = sfft.fft(diff.d2eff)
        diff.fd2veff[:] = sfft.fft(diff.d2veff)

        # Fourier space multiplications
        diff.fd2eff_conv[:] = diff.fd2eff[:] * self.fw2[:]
        diff.fd3_conv[:] = diff.fd3[:] * self.fw3[:]
        diff.fd2veff_conv[:] = diff.fd2veff[:] * (-1.0 * self.fw2vec[:])

        # Transform from Fourier space to real space
        diff.d3_conv[:] = sfft.ifft(diff.fd3_conv).real
        diff.d2eff_conv[:] = sfft.ifft(diff.fd2eff_conv).real
        diff.d2veff_conv[:] = sfft.ifft(diff.fd2veff_conv).real

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
        self.frho_disp = np.zeros(N, dtype=np.cdouble)
        self.fw_rho_disp = np.zeros(N, dtype=np.cdouble)
        self.fmu_disp = np.zeros(N, dtype=np.cdouble)
        self.fw_mu_disp = np.zeros(N, dtype=np.cdouble)
        # Weigthing distance for disperesion term
        self.phi_disp = phi_disp

        planar_weights.__init__(self, dr, R, N, quad)

        # Fourier transformation objects. Allocated in separate method.
        self.fftw_rho_disp = None
        self.ifftw_rho_disp = None
        self.fftw_mu_disp = None
        self.ifftw_mu_disp = None

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

    def convolutions(self, densities: weighted_densities_pc_saft_1D, rho: np.ndarray):
        """

        Args:
            densities:
            rho (np.ndarray): Density profile

        """
        planar_weights.convolutions(self, densities, rho)

        self.frho_disp[:] = sfft.fft(rho)
        # Dispersion density
        self.fw_rho_disp[:] = self.frho_disp[:] * self.fw_disp[:]
        densities.rho_disp[:] = sfft.ifft(self.fw_rho_disp).real

    def correlation_convolution(self, diff: differentials_pc_saft_1D):
        """

        Args:
            diff: Functional differentials

        Returns:

        """
        planar_weights.correlation_convolution(self, diff)


        # Fourier transform derivatives
        self.fmu_disp[:] = sfft.fft(diff.mu_disp)

        # Fourier space multiplications
        self.fw_mu_disp[:] = self.fmu_disp[:] * self.fw_disp[:]

        # Transform from Fourier space to real space
        diff.mu_disp_conv[:] = sfft.ifft(self.fw_mu_disp).real

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
