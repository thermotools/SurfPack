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
        f = np.zeros(dens.N)
        f[:] = -dens.n0[:] * dens.logn3neg[:] + \
               (dens.n1[:] * dens.n2[:] - dens.n1v[:] * dens.n2v[:]) / dens.n3neg[:] + \
               ((dens.n2[:] ** 3) - 3.0 * dens.n2[:] * dens.n2v[:] ** 2) / (24.0 * np.pi * dens.n3neg[:] ** 2)

        return f

    def compressibility(self, eta):
        """
        Calculates the Percus-Yevick HS compressibility from the packing fraction.
        Multiply by rho*kB*T to get pressure

        Args:
        eta (float): Packing fractions

        Returns:
        float: compressibility
        """
        z = (1 + eta + eta ** 2) / (1.0 - eta) ** 3
        return z

    def excess_chemical_potential(self, eta):
        """
        Calculates the reduced HS excess chemical potential from the packing fraction

        Args:
        eta (float): Packing fraction

        Returns:
        float: Excess reduced HS chemical potential ()

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
        diff.d0[:] = -np.log(dens.n3neg[:])
        diff.d1[:] = dens.n2[:] / dens.n3neg[:]
        diff.d2[:] = dens.n1[:] / dens.n3neg[:] + (dens.n2[:] ** 2 - dens.n2v2[:]) / (8 * np.pi * dens.n3neg2[:])
        diff.d3[:] = dens.n0[:] / dens.n3neg[:] + (dens.n1[:] * dens.n2[:] - dens.n1v[:] * dens.n2v[:]) / \
                     dens.n3neg2[:] + (dens.n2[:] ** 3 - 3 * dens.n2[:] * dens.n2v2[:]) / \
                     (12 * np.pi * dens.n3neg[:] ** 3)
        diff.d1v[:] = -dens.n2v[:] / dens.n3neg[:]
        diff.d2v[:] = -(dens.n1v[:] / dens.n3neg[:] + dens.n2[:] * dens.n2v[:] / (4 * np.pi * dens.n3neg2[:]))

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
            dens0.update_utility_variables()
            F1 = self.excess_free_energy(dens0)
            dens0.set_density(i, ni0 + dni)
            dens0.update_utility_variables()
            F2 = self.excess_free_energy(dens0)
            dens0.set_density(i, ni0)
            dens0.update_utility_variables()
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
        # Avoid dividing with zero value of n3 in boundary grid points
        pn3m = dens.n3 > 0.0  # Positive value n3 mask
        f = np.zeros(dens.N)
        f[pn3m] = -dens.n0[pn3m] * dens.logn3neg[pn3m] + \
                  (dens.n1[pn3m] * dens.n2[pn3m] - dens.n1v[pn3m] * dens.n2v[pn3m]) / dens.n3neg[pn3m] + \
                  ((dens.n2[pn3m] ** 3) - 3.0 * dens.n2[pn3m] * dens.n2v[pn3m] ** 2) * \
                  (dens.n3[pn3m] + dens.n3neg2[pn3m] * dens.logn3neg[pn3m]) / \
                  (36.0 * np.pi * dens.n3[pn3m] ** 2 * dens.n3neg2[pn3m])

        return f

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
        if self.numerator is None or np.shape(self.numerator) != np.shape(dens.n0):
            self.numerator = np.zeros_like(dens.n0)
        self.numerator[:] = dens.n3[:] + dens.n3neg2[:] * dens.logn3neg[:]
        if self.denumerator is None or np.shape(self.denumerator) != np.shape(dens.n0):
            self.denumerator = np.zeros_like(dens.n0)
        self.denumerator[:] = (36.0 * np.pi * dens.n3[:] ** 2 * dens.n3neg2[:])

        # Avoid dividing with zero value of n3 in boundary grid points
        pn3m = dens.n3 > 0.0  # Positive value n3 mask
        non_pn3m = np.invert(pn3m)  # Mask for zero and negative value of n3

        diff.d0[pn3m] = -dens.logn3neg[pn3m]
        diff.d1[pn3m] = dens.n2[pn3m] / dens.n3neg[pn3m]
        diff.d2[pn3m] = dens.n1[pn3m] / dens.n3neg[pn3m] + 3 * (dens.n2[pn3m] ** 2 - dens.n2v2[pn3m]) * \
                        self.numerator[pn3m] / self.denumerator[pn3m]
        diff.d3[pn3m] = dens.n0[pn3m] / dens.n3neg[pn3m] + \
                        (dens.n1[pn3m] * dens.n2[pn3m] - dens.n1v[pn3m] * dens.n2v[pn3m]) / dens.n3neg2[pn3m] + \
                        (dens.n2[pn3m] ** 3 - 3 * dens.n2[pn3m] * dens.n2v2[pn3m]) * \
                        ((dens.n3[pn3m] * (5 - dens.n3[pn3m]) - 2) /
                         (self.denumerator[pn3m] * dens.n3neg[pn3m]) - dens.logn3neg[pn3m] / (
                                 18 * np.pi * dens.n3[pn3m] ** 3))
        diff.d1v[pn3m] = -dens.n2v[pn3m] / dens.n3neg[pn3m]
        diff.d2v[pn3m] = -dens.n1v[pn3m] / dens.n3neg[pn3m] - 6 * dens.n2[pn3m] * dens.n2v[pn3m] * \
                         self.numerator[pn3m] / self.denumerator[pn3m]

        # Combining differentials
        diff.combine_differentials()

        # Set non positive n3 grid points to zero
        diff.d3[non_pn3m] = 0.0
        diff.d2eff[non_pn3m] = 0.0
        diff.d2veff[non_pn3m] = 0.0

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

    def update_phi2_and_phi3(self, dens, mask=None):
        """
        Calculates function Phi2 from n3

        Args:
        dens (array_like): Weighted densities
        mask (np.ndarray boolean): Mask for updating phi2 and phi3
        """
        if self.phi2_div3 is None or np.shape(self.phi2_div3) != np.shape(dens.n3):
            self.phi2_div3 = np.zeros_like(dens.n3)
        if self.dphi2dn3_div3 is None or np.shape(self.dphi2dn3_div3) != np.shape(dens.n3):
            self.dphi2dn3_div3 = np.zeros_like(dens.n3)
        if self.phi3_div3 is None or np.shape(self.phi3_div3) != np.shape(dens.n3):
            self.phi3_div3 = np.zeros_like(dens.n3)
        if self.dphi3dn3_div3 is None or np.shape(self.dphi3dn3_div3) != np.shape(dens.n3):
            self.dphi3dn3_div3 = np.zeros_like(dens.n3)
        if mask is None:
            mask = np.full(dens.N, True, dtype=bool)
        prefac = 1.0 / 3.0
        self.phi2_div3[mask] = prefac * (2 - dens.n3[mask] + 2 * dens.n3neg[mask] * dens.logn3neg[mask] / dens.n3[mask])
        self.dphi2dn3_div3[mask] = prefac * (- 1 - 2 / dens.n3[mask] - 2 * dens.logn3neg[mask] / dens.n32[mask])
        self.phi3_div3[mask] = prefac * (
                2 * dens.n3[mask] - 3 * dens.n32[mask] + 2 * dens.n3[mask] * dens.n32[mask] +
                2 * dens.n3neg2[mask] * dens.logn3neg[mask]) / dens.n32[mask]
        self.dphi3dn3_div3[mask] = prefac * (
                - 4 * dens.logn3neg[mask] * dens.n3neg[mask] / (dens.n32[mask] * dens.n3[mask])
                - 4 / dens.n32[mask] + 2 / dens.n3[mask] + 2)

    def excess_free_energy(self, dens):
        """
        Calculates the excess HS Helmholtz free energy from the weighted densities

        Args:
        dens (array_like): Weighted densities

        Returns:
        array_like: Excess HS Helmholtz free energy ()

        """
        # Avoid dividing with zero value of n3 in boundary grid points
        pn3m = dens.n3 > 0.0  # Positive value n3 mask
        self.update_phi2_and_phi3(dens, pn3m)
        f = np.zeros(dens.N)
        f[pn3m] = -dens.n0[pn3m] * dens.logn3neg[pn3m] + \
                  (dens.n1[pn3m] * dens.n2[pn3m] - dens.n1v[pn3m] * dens.n2v[pn3m]) * \
                  (1 + self.phi2_div3) / (dens.n3neg[pn3m]) + \
                  ((dens.n2[pn3m] ** 3) - 3.0 * dens.n2[pn3m] * dens.n2v[pn3m] ** 2) * \
                  (1 - self.phi3_div3[pn3m]) / (24.0 * np.pi * dens.n3neg2[pn3m])

        return f

    def differentials(self, dens, diff):
        """
        Calculates the functional differentials wrpt. the weighted densities

        Args:
        dens (array_like): weighted densities
        diff (array_like): Functional differentials

        """
        # Avoid dividing with zero value of n3 in boundary grid points
        pn3m = dens.n3 > 0.0  # Positive value n3 mask
        non_pn3m = np.invert(pn3m)  # Mask for zero and negative value of n3
        self.update_phi2_and_phi3(dens, pn3m)
        if self.numerator is None or np.shape(self.numerator) != np.shape(dens.n0):
            self.numerator = np.zeros_like(dens.n0)
        self.numerator[:] = 1 - self.phi3_div3[:]
        if self.denumerator is None or np.shape(self.denumerator) != np.shape(dens.n0):
            self.denumerator = np.zeros_like(dens.n0)
        self.denumerator[:] = (24.0 * np.pi * dens.n3neg2[:])

        diff.d0[pn3m] = -dens.logn3neg[pn3m]
        diff.d1[pn3m] = dens.n2[pn3m] * (1 + self.phi2_div3[pn3m]) / dens.n3neg[pn3m]
        diff.d2[pn3m] = dens.n1[pn3m] * (1 + self.phi2_div3[pn3m]) / dens.n3neg[pn3m] + 3 * (
                dens.n2[pn3m] ** 2 - dens.n2v2[pn3m]) * self.numerator[pn3m] / self.denumerator[pn3m]
        diff.d3[pn3m] = dens.n0[pn3m] / dens.n3neg[pn3m] + \
                        (dens.n1[pn3m] * dens.n2[pn3m] - dens.n1v[pn3m] * dens.n2v[pn3m]) * \
                        ((1 + self.phi2_div3[pn3m]) / dens.n3neg2[pn3m] +
                         self.dphi2dn3_div3[pn3m] / dens.n3neg[pn3m]) + \
                        (dens.n2[pn3m] ** 3 - 3 * dens.n2[pn3m] * dens.n2v2[pn3m]) / self.denumerator[pn3m] * \
                        (-self.dphi3dn3_div3[pn3m] + 2 * self.numerator[pn3m] / dens.n3neg[pn3m])
        diff.d1v[pn3m] = -dens.n2v[pn3m] * (1 + self.phi2_div3[pn3m]) / dens.n3neg[pn3m]
        diff.d2v[pn3m] = -dens.n1v[pn3m] * (1 + self.phi2_div3[pn3m]) / dens.n3neg[pn3m] \
                         - 6 * dens.n2[pn3m] * dens.n2v[pn3m] * self.numerator[pn3m] / self.denumerator[pn3m]

        # Combining differentials
        diff.combine_differentials()

        # Set non positive n3 grid points to zero
        diff.d3[non_pn3m] = 0.0
        diff.d2eff[non_pn3m] = 0.0
        diff.d2veff[non_pn3m] = 0.0
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
    dens.print(print_utilities=True)
    RF_functional = Rosenfeld()
    RF_functional.test_differentials(dens)
    WB_functional = WhitebearPureFluid()
    WB_functional.test_differentials(dens)
    WBII_functional = WhitebearMarkIIPureFluid()
    WBII_functional.test_differentials(dens)

