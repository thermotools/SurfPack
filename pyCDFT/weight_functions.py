#!/usr/bin/env python3
import numpy as np
from constants import CONVOLUTIONS, CONV_NO_FFT, CONV_FFTW, CONV_SCIPY_FFT
from scipy.ndimage import convolve1d
from utility import weighted_densities_1D, differentials_1D, \
    allocate_fourier_convolution_variable, allocate_real_convolution_variable
import matplotlib.pyplot as plt
import pyfftw as fftw
import scipy.fft as sfft


class quadrature():
    """
    """

    def __init__(self, NinP):
        """

        Args:
            NinP (int): Number of grid-point inside particle
        """
        self.N = NinP
        self.weights = None

    def get_quadrature_weights(self, quad="Roth"):
        """
        Args:
            quad (str): Name of quadrature (Roth (default), Trapeze/Trapezoidal, None)

        Returns:
            weights (np.ndarray):
        """
        if self.N == 3:
            self.set_simpsons_weights()
        elif quad.upper() == "NONE":
            self.weights = np.ones(self.N)
        elif quad.upper() == "ROTH":
            self.set_roth_weights()
        elif quad.upper() in ("TRAPEZE", "TRAPEZOIDAL"):
            self.set_trapezoidal_weights()
        else:
            raise ValueError("Unknown quadrature: " + quad)

        return self.weights

    def set_simpsons_weights(self):
        """ Small test system weights
        """
        assert self.N == 3
        self.weights = np.ones(self.N)
        self.weights[0] = 3.0 / 8.0
        self.weights[1] = 4.0 / 3.0
        self.weights[2] = 3.0 / 8.0

    def set_roth_weights(self):
        """ Closed extended formula used by Roth 2010
        """
        self.weights = np.ones(self.N)
        assert self.N > 6
        self.weights[0] = 3.0 / 8.0
        self.weights[1] = 7.0 / 6.0
        self.weights[2] = 23.0 / 24.0
        self.weights[-1] = 3.0 / 8.0
        self.weights[-2] = 7.0 / 6.0
        self.weights[-3] = 23.0 / 24.0

    def set_trapezoidal_weights(self):
        """  Trapezoidal weights
        """
        self.weights = np.ones(self.N)
        self.weights[0] = 1.0 / 2.0
        self.weights[-1] = 1.0 / 2.0


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
        NinP = 2 * round(R / dr) + 1
        self.quad = quadrature(NinP)
        self.w3 = np.zeros(NinP)
        self.w2 = np.zeros(NinP)
        self.w2vec = np.zeros(NinP)
        self.x = np.linspace(-self.R, self.R, NinP)
        self.w3[:] = np.pi * (self.R ** 2 - self.x[:] ** 2)
        self.w2[:] = np.pi * np.ones(NinP)[:]
        self.w2vec[:] = 2 * np.pi * self.x[:]

        # Multiply with quadrature weights
        quad_w = self.quad.get_quadrature_weights(quad)
        self.w3 *= quad_w * self.dr
        self.w2 *= quad_w * self.dr
        self.w2vec *= quad_w * self.dr

        # Fourier space variables
        self.fw3 = allocate_fourier_convolution_variable(N)
        self.fw2 = allocate_fourier_convolution_variable(N)
        self.fw2vec = allocate_fourier_convolution_variable(N)
        self.frho = allocate_fourier_convolution_variable(N)

        # Real space variables used for convolution
        self.rho = None

        if CONVOLUTIONS == CONV_FFTW:
            w3_temp = allocate_real_convolution_variable(N)
            w2_temp = allocate_real_convolution_variable(N)
            w2vec_temp = allocate_real_convolution_variable(N)
            NinR = round(R / dr)
            for i in range(NinP):
                j = i - NinR
                w3_temp[j] = self.w3[i]
                w2_temp[j] = self.w2[i]
                w2vec_temp[j] = self.w2vec[i]
            # Calculates weight functions in fourier transforms
            fftw_weights = fftw.FFTW(w3_temp, self.fw3, direction='FFTW_FORWARD', flags=('FFTW_ESTIMATE',))
            fftw_weights.execute()
            fftw_weights.update_arrays(w2_temp, self.fw2)
            fftw_weights.execute()
            fftw_weights.update_arrays(w2vec_temp, self.fw2vec)
            fftw_weights.execute()
            # Delete fftw object and arrays
            del fftw_weights
            del w3_temp
            del w2_temp
            del w2vec_temp
            # Real space variables used for convolution
            self.rho = allocate_real_convolution_variable(N)
        elif CONVOLUTIONS == CONV_SCIPY_FFT:
            w3_temp = allocate_real_convolution_variable(N)
            w2_temp = allocate_real_convolution_variable(N)
            w2vec_temp = allocate_real_convolution_variable(N)
            NinR = round(R / dr)
            for i in range(NinP):
                j = i - NinR
                w3_temp[j] = self.w3[i]
                w2_temp[j] = self.w2[i]
                w2vec_temp[j] = self.w2vec[i]
            self.fw2[:] = sfft.fft(w2_temp)
            self.fw3[:] = sfft.fft(w3_temp)
            self.fw2vec[:] = sfft.fft(w2vec_temp)

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

        Returns:

        """
        if CONVOLUTIONS == CONV_NO_FFT:
            densities.n3[:] = convolve1d(rho, weights=self.w3, mode='nearest')
            densities.n2[:] = convolve1d(rho, weights=self.w2, mode='nearest')
            densities.n2v[:] = convolve1d(rho, weights=self.w2vec, mode='nearest')
        elif CONVOLUTIONS == CONV_FFTW:
            self.rho[:] = rho[:]
            self.fftw_rho()
            # 2d weighted density
            densities.fn2[:] = self.frho[:] * self.fw2[:]
            self.ifftw_n2()

            # 3d weighted density
            densities.fn3[:] = self.frho[:] * self.fw3[:]
            self.ifftw_n3()

            # Vector 2d weighted density
            densities.fn2v[:] = self.frho[:] * self.fw2vec[:]
            self.ifftw_n2v()
        elif CONVOLUTIONS == CONV_SCIPY_FFT:
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
        if CONVOLUTIONS == CONV_NO_FFT:
            densities.n3[:] = convolve1d(rho, weights=self.w3, mode='nearest')
        elif CONVOLUTIONS == CONV_FFTW:
            self.fftw_rho()
            # 3d weighted density
            densities.fn3[:] = self.frho[:] * self.fw3[:]
            self.ifftw_n3()
        elif CONVOLUTIONS == CONV_SCIPY_FFT:
            self.frho[:] = sfft.fft(rho)
            # 3d weighted density
            densities.fn3[:] = self.frho[:] * self.fw3[:]
            densities.n3[:] = sfft.ifft(densities.fn3).real

    def setup_fft(self, densities: weighted_densities_1D, diff: differentials_1D):
        """
        Args:
            densities: Weighted densities
            diff: Functional differentials'

        """
        if CONVOLUTIONS == CONV_FFTW:
            # FFTW objects to perform the fourier transforms
            self.fftw_rho = fftw.FFTW(self.rho, self.frho,
                                      direction='FFTW_FORWARD',
                                      flags=('FFTW_ESTIMATE',))

            self.ifftw_n2 = fftw.FFTW(densities.fn2, densities.n2,
                                      direction='FFTW_BACKWARD',
                                      flags=('FFTW_ESTIMATE',))
            self.ifftw_n3 = fftw.FFTW(densities.fn3, densities.n3,
                                      direction='FFTW_BACKWARD',
                                      flags=('FFTW_ESTIMATE',))
            self.ifftw_n2v = fftw.FFTW(densities.fn2v, densities.n2v,
                                       direction='FFTW_BACKWARD',
                                       flags=('FFTW_ESTIMATE',))

            self.fftw_d2eff = fftw.FFTW(diff.d2eff, diff.fd2eff,
                                        direction='FFTW_FORWARD',
                                        flags=('FFTW_ESTIMATE',))
            self.fftw_d3 = fftw.FFTW(diff.d3, diff.fd3,
                                     direction='FFTW_FORWARD',
                                     flags=('FFTW_ESTIMATE',))
            self.fftw_d2veff = fftw.FFTW(diff.d2veff, diff.fd2veff,
                                         direction='FFTW_FORWARD',
                                         flags=('FFTW_ESTIMATE',))

            self.ifftw_d2eff = fftw.FFTW(diff.fd2eff_conv, diff.d2eff_conv,
                                         direction='FFTW_BACKWARD',
                                         flags=('FFTW_ESTIMATE',))
            self.ifftw_d3 = fftw.FFTW(diff.fd3_conv, diff.d3_conv,
                                      direction='FFTW_BACKWARD',
                                      flags=('FFTW_ESTIMATE',))
            self.ifftw_d2veff = fftw.FFTW(diff.fd2veff_conv, diff.d2veff_conv,
                                          direction='FFTW_BACKWARD',
                                          flags=('FFTW_ESTIMATE',))

    def correlation_convolution(self, diff: differentials_1D):
        """

        Args:
            diff: Functional differentials

        Returns:

        """
        if CONVOLUTIONS == CONV_NO_FFT:
            diff.d3_conv[:] = convolve1d(diff.d3, weights=self.w3, mode='nearest')
            diff.d2eff_conv[:] = convolve1d(diff.d2eff, weights=self.w2, mode='nearest')
            diff.d2veff_conv[:] = -convolve1d(diff.d2veff, weights=self.w2vec, mode='nearest')
        elif CONVOLUTIONS == CONV_FFTW:
            # Fourier transform derivatives
            self.fftw_d2eff()
            self.fftw_d3()
            self.fftw_d2veff()

            # Fourier space multiplications
            diff.fd2eff_conv[:] = diff.fd2eff[:] * self.fw2[:]
            diff.fd3_conv[:] = diff.fd3[:] * self.fw3[:]
            diff.fd2veff_conv[:] = diff.fd2veff[:] * (-1.0 * self.fw2vec[:])

            # Transform from Fourier space to real space
            self.ifftw_d2eff()
            self.ifftw_d3()
            self.ifftw_d2veff()
        elif CONVOLUTIONS == CONV_SCIPY_FFT:
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

    def plot_weights(self):
        """
            Plot weight functions
        """
        R = 0.5
        n_points = 200
        x = np.linspace(-R, R, n_points)
        w3 = np.pi * (R ** 2 - x ** 2)
        w2 = np.pi * np.ones(n_points)
        w2vec = 2 * np.pi * x
        fig = plt.figure()
        plt.plot(x, w3, label="$w_3$")
        plt.plot(x, w2, label="$w_2$")
        plt.plot(x, w2vec, label=r"$w_{2\rm{v}}$")
        leg = plt.legend(loc="best", numpoints=1)
        leg.get_frame().set_linewidth(0.0)
        plt.xlim([-0.51, 0.51])
        plt.xticks([-0.5, -0.25, 0.0, 0.25, 0.5])
        plt.show()

    def plot_actual_weights(self):
        """
            Plot actual weights. Weights multiplied by quadrature
        """
        fig = plt.figure()
        plt.plot(self.x, self.w3, label="$w_3$")
        plt.plot(self.x, self.w2, label="$w_2$")
        plt.plot(self.x, self.w2vec, label=r"$w_{2\rm{v}}$")
        leg = plt.legend(loc="best", numpoints=1)
        leg.get_frame().set_linewidth(0.0)
        plt.xlim([-0.51, 0.51])
        plt.xticks([-0.5, -0.25, 0.0, 0.25, 0.5])
        plt.show()

    def print(self):
        """
            Print weights multiplied by quadrature
        """
        print("w2", self.w2)
        print("w3", self.w3)
        print("w2c", self.w2vec)
        if self.fw2 is not None:
            print("fw2", self.fw2)
        if self.fw3 is not None:
            print("fw3", self.fw3)
        if self.fw2vec is not None:
            print("fw2vec", self.fw2vec)


if __name__ == "__main__":
    plw = planar_weights(dr=0.01, R=0.5, N=200)
    plw.plot_weights()
    plw.plot_actual_weights()
