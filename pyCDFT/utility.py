#!/usr/bin/env python3
import numpy as np


def density_from_packing_fraction(eta, d=1.0):
    """
    Calculates the reduced density of hard-sphere fluid from the packing fraction.

    Args:
        eta (float): Hard-sphere packing fraction
        d (float): Reduced hard-sphere diameter
    Returns:
        density(float): reduced density
    """

    density = 6 * eta / (np.pi * d ** 3)

    return density


def packing_fraction_from_density(density, d=1.0):
    """
    Calculates the hard-sphere fluid packing fraction from reduced density.

    Args:
        density(float): reduced density
        d (float): Reduced hard-sphere diameter
    Returns:
        eta (float): Hard-sphere packing fraction
    """

    eta = density * (np.pi * d ** 3) / 6

    return eta


class weighted_densities_1D():
    """
    """

    def __init__(self, N, R):
        """
        """
        self.N = N
        self.R = R
        self.n0 = np.zeros(N)
        self.n1 = np.zeros(N)
        self.n2 = np.zeros(N)
        self.n3 = np.zeros(N)
        self.n1v = np.zeros(N)
        self.n2v = np.zeros(N)
        # Utilities
        self.n3neg = np.zeros(N)
        self.n3neg2 = np.zeros(N)
        self.n2v2 = np.zeros(N)
        self.logn3neg = np.zeros(N)
        self.n32 = np.zeros(N)

    def update_after_convolution(self):
        self.n1v = self.n2v / (4 * np.pi * self.R)
        self.n0 = self.n2 / (4 * np.pi * self.R ** 2)
        self.n1 = self.n2 / (4 * np.pi * self.R)
        self.n3neg = 1.0 - self.n3
        self.n3neg2 = self.n3neg ** 2
        self.n2v2 = self.n2v ** 2
        self.logn3neg = np.log(self.n3neg)
        self.n32 = self.n3 ** 2

    def set_testing_values(self):
        """
        Set some dummy values for testing differentials
        """
        self.n0 = 1.0
        self.n1 = 2.0
        self.n2 = 3.0
        self.n3 = 0.5
        self.n1v = 5.0
        self.n2v = 6.0

    def get_density(self, i):
        """
        Get weighted density number i
        """
        if i == 0:
            n = self.n0
        elif i == 1:
            n = self.n1
        elif i == 2:
            n = self.n2
        elif i == 3:
            n = self.n3
        elif i == 4:
            n = self.n1v
        elif i == 5:
            n = self.n2v
        else:
            raise ValueError("get_density: Index out of bounds")
        return n

    def set_density(self, i, n):
        """
        Set weighted density number i
        """
        if i == 0:
            self.n0 = n
        elif i == 1:
            self.n1 = n
        elif i == 2:
            self.n2 = n
        elif i == 3:
            self.n3 = n
        elif i == 4:
            self.n1v = n
        elif i == 5:
            self.n2v = n

    def print(self):
        """
        """
        print("\nWeighted densities:")
        print("n0: ", self.n0)
        print("n1: ", self.n1)
        print("n2: ", self.n2)
        print("n3: ", self.n3)
        print("n1v: ", self.n1v)
        print("n2v: ", self.n2v)


class differentials_1D():
    """
    """

    def __init__(self, N, R):
        """
        """
        self.N = N
        self.R = R
        self.d0 = np.zeros(N)
        self.d1 = np.zeros(N)
        self.d2 = np.zeros(N)
        self.d3 = np.zeros(N)
        self.d1v = np.zeros(N)
        self.d2v = np.zeros(N)
        # Utilities
        self.d2eff = np.zeros(N)
        self.d2veff = np.zeros(N)
        # Utilities
        self.d3_conv = np.zeros(N)
        self.d2eff_conv = np.zeros(N)
        self.d2veff_conv = np.zeros(N)
        # One - body direct correlation function
        self.corr = np.zeros(self.N)

    def update_after_convolution(self):
        self.corr = -(self.d3_conv + self.d2eff_conv + self.d2veff_conv)

    def combine_differentials(self):
        """
        Combining differentials to reduce number of convolution integrals
        """
        self.d2eff = self.d0 / (4 * np.pi * self.R ** 2) + self.d1 / (4 * np.pi * self.R) + self.d2
        self.d2veff = self.d1v / (4 * np.pi * self.R) + self.d2v

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

    def print(self):
        """
        """
        print("\nDifferentials")
        print("d0: ", self.d0)
        print("d1: ", self.d1)
        print("d2: ", self.d2)
        print("d3: ", self.d3)
        print("d1v: ", self.d1v)
        print("d2v: ", self.d2v)
        print("d2_eff: ", self.d2eff)
        print("d2v_eff: ", self.d2veff)


if __name__ == "__main__":
    pass
