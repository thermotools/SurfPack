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
from abc import ABC, abstractmethod

class Weights(ABC):
    """
    """

    def __init__(self, dr: float, R: float, N: int):
        """

        Args:
            dr (float): Grid spacing
            R (float): Particle radius
            N (int): Grid size
        """
        self.dr = dr
        self.R = R
        self.N = N
        self.L = self.dr*self.N    # Length

    @abstractmethod
    def convolutions(self, densities: weighted_densities_1D, rho: np.ndarray):
        pass

    @abstractmethod
    def convolution_n3(self, densities: weighted_densities_1D, rho: np.ndarray):
        pass

    @abstractmethod
    def correlation_convolution(self, diff: differentials_1D):
        pass

    @abstractmethod
    def analytical_fourier_weigths(self):
        pass

    @abstractmethod
    def convolutions(self, densities: weighted_densities_pc_saft_1D, rho: np.ndarray):
        pass

    @abstractmethod
    def correlation_convolution(self, diff: differentials_pc_saft_1D):
        pass


class planar_weights(Weights):
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

        # Fourier space variables
        self.fw3 = np.zeros(N, dtype=np.cdouble)
        self.fw2 = np.zeros(N, dtype=np.cdouble)
        self.fw2vec = np.zeros(N, dtype=np.cdouble)
        self.frho = np.zeros(N, dtype=np.cdouble)

        # Real space variables used for convolution
        self.rho = None

        # Extract analytics fourier transform of weights functions
        self.analytical_fourier_weigths()

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
        self.frho_disp = np.zeros(N, dtype=np.cdouble)
        self.fw_rho_disp = np.zeros(N, dtype=np.cdouble)
        self.fmu_disp = np.zeros(N, dtype=np.cdouble)
        self.fw_mu_disp = np.zeros(N, dtype=np.cdouble)
        # Weigthing distance for disperesion term
        self.phi_disp = phi_disp

        planar_weights.__init__(self, dr, R, N)

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


if __name__ == "__main__":
    plw = planar_weights(dr=0.25, R=0.5, N=8)
    plw.plot_weights()
    plw.plot_actual_weights()
