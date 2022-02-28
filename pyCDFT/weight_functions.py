#!/usr/bin/env python3
import numpy as np
from constants import CONVOLUTIONS, CONVNOFFT
from scipy.ndimage import convolve1d
from utility import weighted_densities_1D, differentials_1D
import matplotlib.pyplot as plt


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

    def __init__(self, dr: float, R: float, quad="Roth"):
        """

        Args:
            dr (float): Grid spacing
            R (float): Particle radius
            quad (str): Quadrature for integral
        """
        self.dr = dr
        self.R = R
        NinP = 2 * round(R / dr) + 1
        self.quad = quadrature(NinP)
        print("NinP", NinP, R, dr)
        if CONVOLUTIONS == CONVNOFFT:
            self.w3 = np.zeros(NinP)
            self.w2 = np.zeros(NinP)
            self.w2vec = np.zeros(NinP)
            self.x = np.linspace(-self.R, self.R, NinP)
            self.w3 = np.pi * (self.R ** 2 - self.x ** 2)
            self.w2 = np.pi * np.ones(NinP)
            self.w2vec = 2 * np.pi * self.x

        # Multiply with quadrature weights
        quad_w = self.quad.get_quadrature_weights(quad)
        self.w3 *= quad_w * self.dr
        self.w2 *= quad_w * self.dr
        self.w2vec *= quad_w * self.dr

    def convolutions(self, densities: weighted_densities_1D, rho: np.ndarray):
        """

        Args:
            densities:
            rho (np.ndarray): Density profile

        Returns:

        """
        if CONVOLUTIONS == CONVNOFFT:
            densities.n3 = convolve1d(rho, weights=self.w3, mode='nearest')
            densities.n2 = convolve1d(rho, weights=self.w2, mode='nearest')
            densities.n2v = convolve1d(rho, weights=self.w2vec, mode='nearest')
            densities.update_after_convolution()

    def correlation_convolution(self, diff: differentials_1D):
        """

        Args:
            diff: Functional differentials

        Returns:

        """
        if CONVOLUTIONS == CONVNOFFT:
            diff.d3_conv = convolve1d(diff.d3, weights=self.w3, mode='nearest')
            diff.d2eff_conv = convolve1d(diff.d2eff, weights=self.w2, mode='nearest')
            diff.d2veff_conv = -convolve1d(diff.d2veff, weights=self.w2vec, mode='nearest')
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


if __name__ == "__main__":
    plw = planar_weights(dr=0.01, R=0.5)
    plw.plot_weights()
    plw.plot_actual_weights()
