#!/usr/bin/env python3
import numpy as np
from utility import weighted_densities_1D, differentials_1D


def get_functional(functional="Rosenfeld"):
    """
    """
    functional_name = functional.upper().strip(" -")
    if functional_name in ("ROSENFELD", "RF"):
        func = Rosenfeld()
    elif functional_name in ("WHITEBEAR", "WB"):
        func = WhitebearPureFluid()
    elif functional_name in ("WHITEBEARMARKII", "WHITEBEARII", "WBII"):
        func = WhitebearMarkIIPureFluid()
    else:
        raise ValueError("Unknown functional: " + functional)

    return func


class Rosenfeld:
    """
    Rosenfeld, Yaakov
    Free-energy model for the inhomogeneous hard-sphere fluid mixture and density-functional theory of freezing
    Phys. Rev. Lett. 1989, 63(9):980-983
    doi:10.1103/PhysRevLett.63.980
    """

    def __init__(self):
        """
        No init required
        """
        self.name = "Rosenfeld"
        self.short_name = "RF"

    def excess_free_energy(self, dens):
        """
        Calculates the excess HS Helmholtz free energy from the weighted densities

        Args:
            dens (array_like): Weighted densities

        Returns:
            array_like: Excess HS Helmholtz free energy ()

        """

        dens.n3neg = 1.0 - dens.n3
        F = -dens.n0 * np.log(dens.n3neg) + \
            (dens.n1 * dens.n2 - dens.n1v * dens.n2v) / (dens.n3neg) + \
            ((dens.n2 ** 3) - 3.0 * dens.n2 * dens.n2v ** 2) / (24.0 * np.pi * dens.n3neg ** 2)

        return F

    def compressibility(self, eta):
        """
        Calculates the Percus-Yevick HS compressibility from the packing fraction.
        Multiply by rho*kB*T to get pressure

        Args:
        eta (array_like): Packing fractions

        Returns:
        float: compressibility
        """
        z = (1 + eta + eta ** 2) / (1.0 - eta) ** 3
        return z

    def excess_chemical_potential(self, eta):
        """
        Calculates the reduced HS excess chemical potential from the packing fraction

        Args:
        eta (array_like): Packing fraction

        Returns:
        array_like: Excess reduced HS chemical potential ()

        """
        mu = (14.0 * eta - 13.0 * eta * eta + 5.0 * eta ** 3) / (2.0 * (1 - eta) ** 3)
        mu -= np.log(1.0 - eta)

        return mu

    def differentials(self, dens, diff):
        """
        Calculates the functional differentials wrpt. the weighted densities

        Args:
        dens (array_like): weighted densities
        diff (array_like): Functional differentials

        """
        dens.n3neg = 1.0 - dens.n3
        dens.n3neg2 = dens.n3neg ** 2
        dens.n2v2 = dens.n2v ** 2

        diff.d0 = -np.log(dens.n3neg)
        diff.d1 = dens.n2 / dens.n3neg
        diff.d2 = dens.n1 / dens.n3neg + (dens.n2 ** 2 - dens.n2v2) / (8 * np.pi * dens.n3neg2)
        diff.d3 = dens.n0 / dens.n3neg + (dens.n1 * dens.n2 - dens.n1v * dens.n2v) / (dens.n3neg2) + \
                  (dens.n2 ** 3 - 3 * dens.n2 * dens.n2v2) / (12 * np.pi * dens.n3neg ** 3)
        diff.d1v = -dens.n2v / dens.n3neg
        diff.d2v = -(dens.n1v / dens.n3neg + dens.n2 * dens.n2v / (4 * np.pi * dens.n3neg2))

        # Combining differentials
        diff.combine_differentials()

        return diff

    def test_differentials(self, dens0):
        print("Testing functional " + self.name)
        dFdn = differentials_1D(1, 1.0)
        dFdn = self.differentials(dens, dFdn)
        eps = 1.0e-5
        for i in range(6):
            ni0 = dens0.get_density(i)
            dni = eps * ni0
            dens0.set_density(i, ni0 - dni)
            F1 = self.excess_free_energy(dens0)
            dens0.set_density(i, ni0 + dni)
            F2 = self.excess_free_energy(dens0)
            dens0.set_density(i, ni0)
            dFdn_num = (F2 - F1) / (2 * dni)
            print("Differential: ", i, dFdn_num, dFdn.get_differential(i))


class WhitebearPureFluid(Rosenfeld):
    """
    R. Roth, R. Evans, A. Lang and G. Kahl
    Fundamental measure theory for hard-sphere mixtures revisited: the White Bear version
    Journal of Physics: Condensed Matter
    2002, 14(46):12063-12078
    doi: 10.1088/0953-8984/14/46/313
    """

    def __init__(self):
        """
        """
        super(WhitebearPureFluid, self).__init__()
        self.name = "White Bear"
        self.short_name = "WB"
        self.numerator = None
        self.denumerator = None

    def excess_free_energy(self, dens):
        """
        Calculates the excess HS Helmholtz free energy from the weighted densities

        Args:
        dens (array_like): Weighted densities

        Returns:
        array_like: Excess HS Helmholtz free energy ()

        """

        dens.n3neg = 1.0 - dens.n3
        dens.n3neg2 = dens.n3neg ** 2
        dens.logn3neg = np.log(dens.n3neg)
        F = -dens.n0 * dens.logn3neg + \
            (dens.n1 * dens.n2 - dens.n1v * dens.n2v) / (dens.n3neg) + \
            ((dens.n2 ** 3) - 3.0 * dens.n2 * dens.n2v ** 2) * \
            (dens.n3 + dens.n3neg2 * dens.logn3neg) / (36.0 * np.pi * dens.n3 ** 2 * dens.n3neg2)

        return F

    def compressibility(self, eta):
        """
        Calculates the Carnahan and Starling compressibility from the packing fraction.
        Multiply by rho*kB*T to get pressure

        Carnahan and Starling, doi: 10/dqntps

        Args:
        eta (float): Packing fractions

        Returns:
        float: compressibility

        """
        z = (1 + eta + eta ** 2 - eta ** 3) / (1 - eta) ** 3
        return z

    def excess_chemical_potential(self, eta):
        """
        Calculates the reduced HS excess chemical potential from the packing fraction

        Args:
        eta (array_like): Packing fraction

        Returns:
        array_like: Excess reduced HS chemical potential ()

        """
        mu = (8.0 * eta - 9.0 * eta ** 2 + 3.0 * eta ** 3) / ((1.0 - eta) ** 3)
        return mu

    def differentials(self, dens, diff):
        """
        Calculates the functional differentials wrpt. the weighted densities

        Args:
        dens (array_like): weighted densities
        diff (array_like): Functional differentials

        """
        dens.n3neg = 1.0 - dens.n3
        dens.n3neg2 = dens.n3neg ** 2
        dens.n2v2 = dens.n2v ** 2
        dens.logn3neg = np.log(dens.n3neg)
        if self.numerator is None or np.shape(self.numerator) != np.shape(dens.n0):
            self.numerator = np.zeros_like(dens.n0)
        self.numerator = dens.n3 + dens.n3neg2 * dens.logn3neg
        if self.denumerator is None or np.shape(self.denumerator) != np.shape(dens.n0):
            self.denumerator = np.zeros_like(dens.n0)
        self.denumerator = (36.0 * np.pi * dens.n3 ** 2 * dens.n3neg2)

        diff.d0 = -dens.logn3neg
        diff.d1 = dens.n2 / dens.n3neg
        diff.d2 = dens.n1 / dens.n3neg + 3 * (dens.n2 ** 2 - dens.n2v2) * self.numerator / self.denumerator
        diff.d3 = dens.n0 / dens.n3neg + (dens.n1 * dens.n2 - dens.n1v * dens.n2v) / dens.n3neg2 + \
                  (dens.n2 ** 3 - 3 * dens.n2 * dens.n2v2) * \
                  ((dens.n3 * (5 - dens.n3) - 2) / (self.denumerator * dens.n3neg) - dens.logn3neg / (
                          18 * np.pi * dens.n3 ** 3))
        diff.d1v = -dens.n2v / dens.n3neg
        diff.d2v = -dens.n1v / dens.n3neg - 6 * dens.n2 * dens.n2v * self.numerator / self.denumerator

        # Combining differentials
        diff.combine_differentials()

        return diff


class WhitebearMarkIIPureFluid(WhitebearPureFluid):
    """
    Hendrik Hansen-Goos and Roland Roth
    Density functional theory for hard-sphere mixtures: the White Bear version mark II
    Journal of Physics: Condensed Matter
    2006, 18(37): 8413-8425
    doi: 10.1088/0953-8984/18/37/002
    """

    def __init__(self):
        """
        """
        super(WhitebearMarkIIPureFluid, self).__init__()
        self.name = "White Bear Mark II"
        self.short_name = "WBII"
        self.phi2_div3 = None
        self.dphi2dn3_div3 = None
        self.phi3_div3 = None
        self.dphi3dn3_div3 = None

    def update_phi2_and_phi3(self, dens):
        """
        Calculates function Phi2 from n3

        Args:
        dens (array_like): Weighted densities
        """
        if self.phi2_div3 is None or np.shape(self.phi2_div3) != np.shape(dens.n3):
            self.phi2_div3 = np.zeros_like(dens.n3)
        if self.dphi2dn3_div3 is None or np.shape(self.dphi2dn3_div3) != np.shape(dens.n3):
            self.dphi2dn3_div3 = np.zeros_like(dens.n3)
        if self.phi3_div3 is None or np.shape(self.phi3_div3) != np.shape(dens.n3):
            self.phi3_div3 = np.zeros_like(dens.n3)
        if self.dphi3dn3_div3 is None or np.shape(self.dphi3dn3_div3) != np.shape(dens.n3):
            self.dphi3dn3_div3 = np.zeros_like(dens.n3)

        prefac = 1.0 / 3.0
        dens.n32 = dens.n3 ** 2
        self.phi2_div3 = prefac * (2 - dens.n3 + 2 * dens.n3neg * dens.logn3neg / dens.n3)
        self.dphi2dn3_div3 = prefac * (- 1 - 2 / dens.n3 - 2 * dens.logn3neg / dens.n32)
        self.phi3_div3 = prefac * (
                2 * dens.n3 - 3 * dens.n32 + 2 * dens.n3 * dens.n32 + 2 * dens.n3neg2 * dens.logn3neg) / dens.n32
        self.dphi3dn3_div3 = prefac * (
                - 4 * dens.logn3neg * dens.n3neg / (dens.n32 * dens.n3) - 4 / dens.n32 + 2 / dens.n3 + 2)

    def excess_free_energy(self, dens):
        """
        Calculates the excess HS Helmholtz free energy from the weighted densities

        Args:
        dens (array_like): Weighted densities

        Returns:
        array_like: Excess HS Helmholtz free energy ()

        """

        dens.n3neg = 1.0 - dens.n3
        dens.n3neg2 = dens.n3neg ** 2
        dens.logn3neg = np.log(dens.n3neg)
        self.update_phi2_and_phi3(dens)
        F = -dens.n0 * dens.logn3neg + \
            (dens.n1 * dens.n2 - dens.n1v * dens.n2v) * (1 + self.phi2_div3) / (dens.n3neg) + \
            ((dens.n2 ** 3) - 3.0 * dens.n2 * dens.n2v ** 2) * \
            (1 - self.phi3_div3) / (24.0 * np.pi * dens.n3 ** 2 * dens.n3neg2)

        return F

    def differentials(self, dens, diff):
        """
        Calculates the functional differentials wrpt. the weighted densities

        Args:
        dens (array_like): weighted densities
        diff (array_like): Functional differentials

        """
        dens.n3neg = 1.0 - dens.n3
        dens.n3neg2 = dens.n3neg ** 2
        dens.n2v2 = dens.n2v ** 2
        dens.logn3neg = np.log(dens.n3neg)
        self.update_phi2_and_phi3(dens)
        if self.numerator is None or np.shape(self.numerator) != np.shape(dens.n0):
            self.numerator = np.zeros_like(dens.n0)
        self.numerator = 1 - self.phi3_div3
        if self.denumerator is None or np.shape(self.denumerator) != np.shape(dens.n0):
            self.denumerator = np.zeros_like(dens.n0)
        self.denumerator = (24.0 * np.pi * dens.n3 ** 2 * dens.n3neg2)

        diff.d0 = -dens.logn3neg
        diff.d1 = dens.n2 * (1 + self.phi2_div3) / dens.n3neg
        diff.d2 = dens.n1 * (1 + self.phi2_div3) / dens.n3neg + 3 * (
                dens.n2 ** 2 - dens.n2v2) * self.numerator / self.denumerator
        diff.d3 = dens.n0 / dens.n3neg + (dens.n1 * dens.n2 - dens.n1v * dens.n2v) * \
                  ((1 + self.phi2_div3) / dens.n3neg2 + self.dphi2dn3_div3 / dens.n3neg) + \
                  (dens.n2 ** 3 - 3 * dens.n2 * dens.n2v2) / self.denumerator * \
                  (-self.dphi3dn3_div3 - 2 * self.numerator / dens.n3 + 2 * self.numerator / dens.n3neg)

        diff.d1v = -dens.n2v * (1 + self.phi2_div3) / dens.n3neg
        diff.d2v = -dens.n1v * (
                1 + self.phi2_div3) / dens.n3neg - 6 * dens.n2 * dens.n2v * self.numerator / self.denumerator

        # Combining differentials
        diff.combine_differentials()

        return diff


class WhitebearMixture(Rosenfeld):
    """
    R. Roth, R. Evans, A. Lang and G. Kahl
    Fundamental measure theory for hard-sphere mixtures revisited: the White Bear version
    Journal of Physics: Condensed Matter
    2002, 14(46):12063-12078
    doi: 10.1088/0953-8984/14/46/313
    """

    def compressibility(self, eta):
        """
        Calculates the Boublik and  Mansoori, Carnahan, Starling, and Leland (BMCSL) compressibility
        from the packing fraction. Multiply by rho*kB*T to get pressure

        T. Boublik, doi: 10/bjgkjg
        G. A. Mansoori, N. F. Carnahan, K. E. Starling, T. Leland, doi: 10/dkfhh7

        Args:
        eta (array_like): Packing fractions

        Returns:
        float: compressibility

        """
        denum = 1 - eta[3]
        z = (eta[0] / denum + 3 * eta[1] * eta[2] / denum ** 2 + (3 - eta[3]) * eta[2] ** 3 / denum ** 3) / eta[0]
        return z


if __name__ == "__main__":
    # Model testing
    dens = weighted_densities_1D(1, 0.5)
    dens.set_testing_values()
    RF_functional = Rosenfeld()
    RF_functional.test_differentials(dens)
    WB_functional = WhitebearPureFluid()
    WB_functional.test_differentials(dens)
    WBII_functional = WhitebearMarkIIPureFluid()
    WBII_functional.test_differentials(dens)
