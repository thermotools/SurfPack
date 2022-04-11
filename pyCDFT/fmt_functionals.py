#!/usr/bin/env python3
import numpy as np
from utility import weighted_densities_1D, get_thermopack_model, \
    weighted_densities_pc_saft_1D
from pyctp import pcsaft
from constants import NA, RGAS


def get_functional(N, T, functional="Rosenfeld", R=np.array([0.5]), thermopack=None):
    """
    Return functional class based on functional name.

    Args:
        N (int): Grid size
        T (float): Reduced temperature
        functional (str): Name of functional
        R (ndarray): Particle radius for all components
        thermopack (thermo): Thermopack object
    """
    functional_name = functional.upper().strip(" -")
    if functional_name in ("ROSENFELD", "RF"):
        func = Rosenfeld(N=N, R=R)
    elif functional_name in ("WHITEBEAR", "WB"):
        func = Whitebear(N=N, R=R)
    elif functional_name in ("WHITEBEARMARKII", "WHITEBEARII", "WBII"):
        func = WhitebearMarkII(N=N, R=R)
    elif functional_name in ("PC-SAFT", "PCSAFT"):
        func = pc_saft(N=N, pcs=thermopack, T_red=T)
    else:
        raise ValueError("Unknown functional: " + functional)

    return func


class bulk_weighted_densities:
    """
    Utility class for calculating bulk states.
    """

    def __init__(self, rho_b, R):
        """

        Args:
            rho_b (ndarray): Bulk densities
            R (ndarray): Particle radius for all components
        """
        self.rho_i = np.array(rho_b)
        self.rho_i[:] = rho_b[:]
        self.n = np.zeros(4)
        self.n[0] = np.sum(rho_b)
        self.n[1] = np.sum(R * rho_b)
        self.n[2] = 4*np.pi*np.sum(R ** 2 * rho_b)
        self.n[3] = 4 * np.pi * np.sum(R ** 3 * rho_b) / 3
        self.dndrho = np.zeros((4, np.shape(rho_b)[0]))
        self.dndrho[0, :] = 1.0
        self.dndrho[1, :] = R
        self.dndrho[2, :] = 4*np.pi*R**2
        self.dndrho[3, :] = 4*np.pi*R**3/3

    def print(self):
        print("Bulk weighted densities:")
        print("n_0: ", self.n[0])
        print("n_1: ", self.n[1])
        print("n_2: ", self.n[2])
        print("n_3: ", self.n[3])


class Rosenfeld:
    """
    Rosenfeld, Yaakov
    Free-energy model for the inhomogeneous hard-sphere fluid mixture andl
    density-functional theory of freezing.
    Phys. Rev. Lett. 1989, 63(9):980-983
    doi:10.1103/PhysRevLett.63.980
    """

    def __init__(self, N, R=np.array([0.5])):
        """

        Args:
            N (integer): Grid size
            R (ndarray): Particle radius for all components
        """
        self.name = "Rosenfeld"
        self.short_name = "RF"
        self.R = R
        self.nc = np.shape(R)[0]
        self.N = N
        # Allocate arrays for differentials
        self.d0 = np.zeros(N)
        self.d1 = np.zeros(N)
        self.d2 = np.zeros(N)
        self.d3 = np.zeros(N)
        self.d1v = np.zeros(N)
        self.d2v = np.zeros(N)
        self.d2eff = np.zeros(N)
        self.d2veff = np.zeros(N)

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
            ((dens.n2[:] ** 3) - 3.0 * dens.n2[:] * dens.n2v[:]
             ** 2) / (24.0 * np.pi * dens.n3neg[:] ** 2)

        return f

    def bulk_compressibility(self, rho_b):
        """
        Calculates the Percus-Yevick HS compressibility from the
        packing fraction. Multiply by rho*kB*T to get pressure.

        Args:
            rho_b (ndarray): Bulk densities

        Returns:
            float: compressibility
        """
        bd = bulk_weighted_densities(rho_b, self.R)
        phi, dphidn = self.bulk_functional_with_differentials(
            bd, only_hs_system=True)
        beta_p_ex = - phi + np.sum(dphidn[:4] * bd.n)
        beta_p_id = bd.n[0]
        z = (beta_p_id + beta_p_ex)/bd.n[0]
        return z

    def bulk_excess_chemical_potential(self, rho_b):
        """
        Calculates the reduced HS excess chemical potential from the bulk
        packing fraction.

        Args:
        rho_b (ndarray): Bulk densities

        Returns:
        float: Excess reduced HS chemical potential ()

        """
        bd = bulk_weighted_densities(rho_b, self.R)
        phi, dphidn = self.bulk_functional_with_differentials(
            bd, only_hs_system=True)
        mu_ex = np.zeros(self.nc)
        for i in range(self.nc):
            mu_ex[i] = np.sum(dphidn[:4] * bd.dndrho[:, i])

        return mu_ex

    def bulk_functional_with_differentials(self, bd, only_hs_system=False):
        """
        Calculates the functional differentials wrpt. the weighted densities
        in the bulk phase.

        Args:
        bd (bulk_weighted_densities): bulk_weighted_densities
        only_hs_system (bool): Only calculate for hs-system

        """
        n3neg = 1.0-bd.n[3]
        d0 = -np.log(n3neg)
        d1 = bd.n[2] / n3neg
        d2 = bd.n[1] / n3neg + bd.n[2] ** 2 / (8 * np.pi * n3neg ** 2)
        d3 = bd.n[0] / n3neg + bd.n[1] * bd.n[2] / n3neg ** 2 \
            + bd.n[2] ** 3 / (12 * np.pi * n3neg ** 3)
        dphidn = np.array([d0, d1, d2, d3])
        phi = d0 * bd.n[0] + d1 * bd.n[1] + \
            bd.n[2] ** 3 / (24 * np.pi * n3neg ** 2)
        return phi, dphidn

    def differentials(self, dens):
        """
        Calculates the functional differentials wrpt. the weighted densities.

        Args:
        dens (array_like): weighted densities
        diff (array_like): Functional differentials

        """
        self.d0[:] = -np.log(dens.n3neg[:])
        self.d1[:] = dens.n2[:] / dens.n3neg[:]
        self.d2[:] = dens.n1[:] / dens.n3neg[:] + \
            (dens.n2[:] ** 2 - dens.n2v2[:]) / (8 * np.pi * dens.n3neg2[:])
        self.d3[:] = dens.n0[:] / dens.n3neg[:] + (dens.n1[:] * dens.n2[:] - dens.n1v[:] * dens.n2v[:]) / \
            dens.n3neg2[:] + (dens.n2[:] ** 3 - 3 * dens.n2[:] * dens.n2v2[:]) / \
            (12 * np.pi * dens.n3neg[:] ** 3)
        self.d1v[:] = -dens.n2v[:] / dens.n3neg[:]
        self.d2v[:] = -(dens.n1v[:] / dens.n3neg[:] + dens.n2[:]
                        * dens.n2v[:] / (4 * np.pi * dens.n3neg2[:]))

        # Combining differentials
        self.combine_differentials()

    def combine_differentials(self):
        """
        Combining differentials to reduce number of convolution integrals
        """
        self.d2eff[:] = self.d0[:] / (4 * np.pi * self.R ** 2) + \
            self.d1[:] / (4 * np.pi * self.R) + self.d2[:]
        self.d2veff[:] = self.d1v[:] / (4 * np.pi * self.R) + self.d2v[:]

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

    def test_differentials(self, dens0):
        """

        Args:
            dens0 (weighted_densities_1D): Weighted densities

        """
        print("Testing functional " + self.name)
        self.differentials(dens0)
        #F0 = self.excess_free_energy(dens0)
        eps = 1.0e-5
        ni0 = np.zeros_like(dens0.n0)
        dni = np.zeros_like(dens0.n0)
        for i in range(dens0.n_max_test):
            ni0[:] = dens0.get_density(i)
            dni[:] = eps * ni0[:]
            dens0.set_density(i, ni0 - dni)
            dens0.update_utility_variables()
            F1 = self.excess_free_energy(dens0)
            dens0.set_density(i, ni0 + dni)
            dens0.update_utility_variables()
            F2 = self.excess_free_energy(dens0)
            dens0.set_density(i, ni0)
            dens0.update_utility_variables()
            dFdn_num = (F2 - F1) / (2 * dni)
            print("Differential: ", i, dFdn_num, self.get_differential(i))

    def test_bulk_differentials(self, rho_b):
        """

        Args:
            rho_b (np.ndarray): Bulk densities

        """
        print("Testing functional " + self.name)
        bd0 = bulk_weighted_densities(rho_b, self.R)
        phi, dphidn = self.bulk_functional_with_differentials(bd0)

        print("HS functional differentials:")
        for i in range(4):
            bd = bulk_weighted_densities(rho_b, self.R)
            eps = 1.0e-5 * bd.n[i]
            bd.n[i] += eps
            phi2, dphidn2 = self.bulk_functional_with_differentials(bd)
            bd = bulk_weighted_densities(rho_b, self.R)
            eps = 1.0e-5 * bd.n[i]
            bd.n[i] -= eps
            phi1, dphidn1 = self.bulk_functional_with_differentials(bd)
            dphidn_num = (phi2 - phi1) / (2 * eps)
            print("Differential: ", i, dphidn_num, dphidn[i])

        mu_ex = self.bulk_excess_chemical_potential(rho_b)
        rho_b_local = np.zeros_like(rho_b)
        print("Functional differentials:")
        for i in range(self.nc):
            eps = 1.0e-5 * rho_b[i]
            rho_b_local[:] = rho_b[:]
            rho_b_local[i] += eps
            bd = bulk_weighted_densities(rho_b_local, self.R)
            phi2, dphidn = self.bulk_functional_with_differentials(bd)
            phi2_hs, _ = self.bulk_functional_with_differentials(
                bd, only_hs_system=True)
            rho_b_local[:] = rho_b[:]
            rho_b_local[i] -= eps
            bd = bulk_weighted_densities(rho_b_local, self.R)
            phi1, dphidn = self.bulk_functional_with_differentials(bd)
            phi1_hs, _ = self.bulk_functional_with_differentials(
                bd, only_hs_system=True)
            dphidrho_num_no_hs = (phi2 - phi2_hs - phi1 + phi1_hs) / (2 * eps)
            dphidrho_num = (phi2 - phi1) / (2 * eps)
            if np.shape(dphidn)[0] > 4:
                print("Differential: ", 4+i, dphidrho_num_no_hs, dphidn[4+i])
            print("Chemical potential comp.: ", i, dphidrho_num, mu_ex[i])


class Whitebear(Rosenfeld):
    """
    R. Roth, R. Evans, A. Lang and G. Kahl
    Fundamental measure theory for hard-sphere mixtures revisited: the White Bear version
    Journal of Physics: Condensed Matter
    2002, 14(46):12063-12078
    doi: 10.1088/0953-8984/14/46/313

    In the bulk phase the functional reduces to the Boublik and
    Mansoori, Carnahan, Starling, and Leland (BMCSL) EOS.
    T. Boublik, doi: 10/bjgkjg
    G. A. Mansoori, N. F. Carnahan, K. E. Starling, T. Leland, doi: 10/dkfhh7

    """

    def __init__(self, N, R=np.array([0.5])):
        """

        Args:
            N (integer): Grid size
            R (ndarray): Particle radius for all components
        """
        super(Whitebear, self).__init__(N, R)
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

    def bulk_functional_with_differentials(self, bd, only_hs_system=False):
        """
        Calculates the functional differentials wrpt. the weighted densities
        in the bulk phase.

        Args:
        bd (bulk_weighted_densities): bulk_weighted_densities
        only_hs_system (bool): Only calculate for hs-system

        """
        n3neg = 1.0-bd.n[3]
        numerator = bd.n[3] + n3neg ** 2 * np.log(n3neg)
        denumerator = 36.0 * np.pi * bd.n[3] ** 2 * n3neg ** 2
        d0 = -np.log(n3neg)
        d1 = bd.n[2] / n3neg
        d2 = bd.n[1] / n3neg + 3 * bd.n[2] ** 2 * numerator / denumerator
        d3 = bd.n[0] / n3neg + \
            bd.n[1] * bd.n[2] / n3neg ** 2 + \
            bd.n[2] ** 3 * \
            ((bd.n[3] * (5 - bd.n[3]) - 2) /
             (denumerator * n3neg) - np.log(n3neg) /
             (18 * np.pi * bd.n[3] ** 3))

        dphidn = np.array([d0, d1, d2, d3])
        phi = d0 * bd.n[0] + d1 * bd.n[1] + \
            bd.n[2] ** 3 * numerator / denumerator
        return phi, dphidn

    def differentials(self, dens):
        """
        Calculates the functional differentials wrpt. the weighted densities

        Args:
        dens (array_like): weighted densities

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

        self.d0[pn3m] = -dens.logn3neg[pn3m]
        self.d1[pn3m] = dens.n2[pn3m] / dens.n3neg[pn3m]
        self.d2[pn3m] = dens.n1[pn3m] / dens.n3neg[pn3m] + 3 * (dens.n2[pn3m] ** 2 - dens.n2v2[pn3m]) * \
            self.numerator[pn3m] / self.denumerator[pn3m]
        self.d3[pn3m] = dens.n0[pn3m] / dens.n3neg[pn3m] + \
            (dens.n1[pn3m] * dens.n2[pn3m] - dens.n1v[pn3m] * dens.n2v[pn3m]) / dens.n3neg2[pn3m] + \
            (dens.n2[pn3m] ** 3 - 3 * dens.n2[pn3m] * dens.n2v2[pn3m]) * \
            ((dens.n3[pn3m] * (5 - dens.n3[pn3m]) - 2) /
             (self.denumerator[pn3m] * dens.n3neg[pn3m]) - dens.logn3neg[pn3m] / (
                18 * np.pi * dens.n3[pn3m] ** 3))
        self.d1v[pn3m] = -dens.n2v[pn3m] / dens.n3neg[pn3m]
        self.d2v[pn3m] = -dens.n1v[pn3m] / dens.n3neg[pn3m] - 6 * dens.n2[pn3m] * dens.n2v[pn3m] * \
            self.numerator[pn3m] / self.denumerator[pn3m]

        # Combining differentials
        self.combine_differentials()

        # Set non positive n3 grid points to zero
        self.d3[non_pn3m] = 0.0
        self.d2eff[non_pn3m] = 0.0
        self.d2veff[non_pn3m] = 0.0


class WhitebearMarkII(Whitebear):
    """
    Hendrik Hansen-Goos and Roland Roth
    Density functional theory for hard-sphere mixtures:
    the White Bear version mark II.
    Journal of Physics: Condensed Matter
    2006, 18(37): 8413-8425
    doi: 10.1088/0953-8984/18/37/002
    """

    def __init__(self, N, R=np.array([0.5])):
        """

        Args:
            R (ndarray): Radius of particles
        """
        super(WhitebearMarkII, self).__init__(N, R)
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
        self.phi2_div3[mask] = prefac * (2 - dens.n3[mask] + 2 *
                                         dens.n3neg[mask] * dens.logn3neg[mask] / dens.n3[mask])
        self.dphi2dn3_div3[mask] = prefac * \
            (- 1 - 2 / dens.n3[mask] - 2 *
             dens.logn3neg[mask] / dens.n32[mask])
        self.phi3_div3[mask] = prefac * (
            2 * dens.n3[mask] - 3 * dens.n32[mask] + 2 * dens.n3[mask] * dens.n32[mask] +
            2 * dens.n3neg2[mask] * dens.logn3neg[mask]) / dens.n32[mask]
        self.dphi3dn3_div3[mask] = prefac * (
            - 4 * dens.logn3neg[mask] * dens.n3neg[mask] /
            (dens.n32[mask] * dens.n3[mask])
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

    def calc_phi2_and_phi3_bulk(self, bd):
        """
        Calculates function  Phi2 and Phi3 from n3

        Args:
        dens (bulk_weighted_densities): Weighted densities
        """

        prefac = 1.0 / 3.0
        n3neg = 1.0-bd.n[3]
        logn3neg = np.log(n3neg)
        phi2_div3 = prefac * (2 - bd.n[3] + 2 *
                              n3neg * logn3neg / bd.n[3])
        dphi2dn3_div3 = prefac * \
            (- 1 - 2 / bd.n[3] - 2 * logn3neg / bd.n[3] ** 2)
        phi3_div3 = prefac * (
            2 * bd.n[3] - 3 * bd.n[3] ** 2 + 2 * bd.n[3] * bd.n[3] ** 2 +
            2 * n3neg ** 2 * logn3neg) / bd.n[3] ** 2
        dphi3dn3_div3 = prefac * \
            (- 4 * logn3neg * n3neg /
             (bd.n[3] ** 2 * bd.n[3]) - 4 / bd.n[3] ** 2 + 2 / bd.n[3] + 2)
        return phi2_div3, dphi2dn3_div3, phi3_div3, dphi3dn3_div3

    def bulk_functional_with_differentials(self, bd, only_hs_system=False):
        """
        Calculates the functional differentials wrpt. the weighted densities
        in the bulk phase.

        Args:
        bd (bulk_weighted_densities): bulk_weighted_densities
        only_hs_system (bool): Only calculate for hs-system

        """
        phi2_div3, dphi2dn3_div3, phi3_div3, dphi3dn3_div3 = \
            self.calc_phi2_and_phi3_bulk(bd)
        n3neg = 1.0-bd.n[3]
        numerator = 1 - phi3_div3
        denumerator = 24.0 * np.pi * n3neg ** 2
        d0 = -np.log(n3neg)
        d1 = bd.n[2] * (1 + phi2_div3) / n3neg
        d2 = bd.n[1] * (1 + phi2_div3) / n3neg + 3 * \
            bd.n[2] ** 2 * numerator / denumerator
        d3 = bd.n[0] / n3neg + \
            bd.n[1] * bd.n[2] * \
            ((1 + phi2_div3) / n3neg ** 2 +
             dphi2dn3_div3 / n3neg) + \
            bd.n[2] ** 3 / denumerator * \
            (-dphi3dn3_div3 + 2 *
             numerator / n3neg)

        dphidn = np.array([d0, d1, d2, d3])
        phi = d0 * bd.n[0] + d1 * bd.n[1] + \
            bd.n[2] ** 3 * numerator / denumerator
        return phi, dphidn

    def differentials(self, dens):
        """
        Calculates the functional differentials wrpt. the weighted densities

        Args:
        dens (array_like): weighted densities

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

        self.d0[pn3m] = -dens.logn3neg[pn3m]
        self.d1[pn3m] = dens.n2[pn3m] * \
            (1 + self.phi2_div3[pn3m]) / dens.n3neg[pn3m]
        self.d2[pn3m] = dens.n1[pn3m] * (1 + self.phi2_div3[pn3m]) / dens.n3neg[pn3m] + 3 * (
            dens.n2[pn3m] ** 2 - dens.n2v2[pn3m]) * self.numerator[pn3m] / self.denumerator[pn3m]
        self.d3[pn3m] = dens.n0[pn3m] / dens.n3neg[pn3m] + \
            (dens.n1[pn3m] * dens.n2[pn3m] - dens.n1v[pn3m] * dens.n2v[pn3m]) * \
            ((1 + self.phi2_div3[pn3m]) / dens.n3neg2[pn3m] +
             self.dphi2dn3_div3[pn3m] / dens.n3neg[pn3m]) + \
            (dens.n2[pn3m] ** 3 - 3 * dens.n2[pn3m] * dens.n2v2[pn3m]) / self.denumerator[pn3m] * \
            (-self.dphi3dn3_div3[pn3m] + 2 *
             self.numerator[pn3m] / dens.n3neg[pn3m])
        self.d1v[pn3m] = -dens.n2v[pn3m] * \
            (1 + self.phi2_div3[pn3m]) / dens.n3neg[pn3m]
        self.d2v[pn3m] = -dens.n1v[pn3m] * (1 + self.phi2_div3[pn3m]) / dens.n3neg[pn3m] \
            - 6 * dens.n2[pn3m] * dens.n2v[pn3m] * \
            self.numerator[pn3m] / self.denumerator[pn3m]

        # Combining differentials
        self.combine_differentials()

        # Set non positive n3 grid points to zero
        self.d3[non_pn3m] = 0.0
        self.d2eff[non_pn3m] = 0.0
        self.d2veff[non_pn3m] = 0.0


class pc_saft(Whitebear):
    """

    """

    def __init__(self, N, pcs: pcsaft, T_red):
        """

        Args:
            pcs (pcsaft): Thermopack object
            T_red (float): Reduced temperature
            R (ndarray): Particle radius for all components
        """
        self.thermo = pcs
        self.T_red = T_red
        self.T = self.T_red * self.thermo.eps_div_kb[0]
        self.d_hs = np.zeros(pcs.nc)
        for i in range(pcs.nc):
            self.d_hs[i] = pcs.hard_sphere_diameters(self.T)
        R = np.zeros(pcs.nc)
        R[:] = 0.5*self.d_hs[:]/self.d_hs[0]
        Whitebear.__init__(self, N, R)
        self.name = "PC-SAFT"
        self.short_name = "PC"
        self.mu_disp = np.zeros((N, pcs.nc))

    def excess_free_energy(self, dens):
        """
        Calculates the excess HS Helmholtz free energy from the weighted densities

        Args:
        dens (array_like): Weighted densities

        Returns:
        array_like: Excess HS Helmholtz free energy ()

        """
        f = Whitebear.excess_free_energy(self, dens)
        rho_thermo = np.zeros(self.nc)
        V = 1.0
        for i in range(len(f)):
            rho_thermo[:] = dens.rho_disp_array[i, :]
            rho_mix = np.sum(rho_thermo)
            rho_thermo *= 1.0/(NA*self.d_hs[0]**3)
            a, = self.thermo.a_dispersion(self.T, V, rho_thermo)
            f[i] += rho_mix*a

        return f

    def differentials(self, dens):
        """
        Calculates the functional differentials wrpt. the weighted densities

        Args:
        dens (array_like): weighted densities
        diff (array_like): Functional differentials

        """
        Whitebear.differentials(self, dens)

        # All densities must be positive
        prdm = dens.rho_disp > 0.0  # Positive rho_disp value mask
        for i in range(self.nc):
            np.logical_and(prdm, dens.rho_disp_array[:, i] > 0.0, out=prdm)
        # Mask for zero and negative value of rho_disp
        non_prdm = np.invert(prdm)
        rho_thermo = np.zeros(self.nc)
        V = 1.0
        for i in range(self.N):
            if prdm[i]:
                rho_thermo[:] = dens.rho_disp_array[i, :]
                rho_thermo *= 1.0/(NA*self.d_hs[0]**3)
                a, a_n, = self.thermo.a_dispersion(
                    self.T, V, rho_thermo, a_n=True)
                self.mu_disp[i, :] = a + np.sum(rho_thermo)*a_n[:]

        self.mu_disp[non_prdm, :] = 0.0

    def bulk_compressibility(self, rho_b):
        """
        Calculates the PC-SAFT compressibility.
        Multiply by rho*kB*T to get pressure.

        Args:
            rho_b (ndarray): Bulk densities

        Returns:
            float: compressibility
        """
        z = Whitebear.bulk_compressibility(self, rho_b)
        # PC-SAFT contributions
        rho_thermo = np.array(rho_b)
        rho_thermo *= 1.0/(NA*self.d_hs[0]**3)
        rho_mix = np.sum(rho_thermo)
        V = 1.0/rho_mix
        n = rho_thermo/rho_mix
        a, a_V, = self.thermo.a_dispersion(
            self.T, V, n, a_v=True)
        z_r = -a_V*V
        z += z_r
        return z

    def bulk_excess_chemical_potential(self, rho_b):
        """
        Calculates the reduced HS excess chemical potential from the bulk
        packing fraction.

        Args:
        rho_b (ndarray): Bulk densities

        Returns:
        float: Excess reduced HS chemical potential ()

        """
        mu_ex = Whitebear.bulk_excess_chemical_potential(self, rho_b)
        # PC-SAFT contributions
        rho_thermo = np.array(rho_b)
        rho_thermo *= 1.0/(NA*self.d_hs[0]**3)
        rho_mix = np.sum(rho_thermo)
        V = 1.0/rho_mix
        n = rho_thermo/rho_mix
        a, a_n, = self.thermo.a_dispersion(
            self.T, V, n, a_n=True)
        a_n += a
        mu_ex += a_n
        return mu_ex

    def bulk_functional_with_differentials(self, bd, only_hs_system=False):
        """
        Calculates the functional differentials wrpt. the weighted densities
        in the bulk phase.

        Args:
        bd (bulk_weighted_densities): bulk_weighted_densities
        only_hs_system (bool): Only calculate for hs-system
        """
        phi, dphidn = Whitebear.bulk_functional_with_differentials(self, bd)
        if not only_hs_system:
            rho_vec = bd.rho_i
            rho_mix = np.sum(rho_vec)
            V = 1.0
            rho_thermo = rho_vec/(NA*self.d_hs[0]**3)
            a, a_n, = self.thermo.a_dispersion(
                self.T, V, rho_thermo, a_n=True)
            phi += rho_mix*a
            dphidn_comb = np.zeros(4 + self.nc)
            dphidn_comb[:4] = dphidn
            dphidn_comb[4:] = a + np.sum(rho_thermo)*a_n[:]
        else:
            dphidn_comb = dphidn
        return phi, dphidn_comb

    def get_differential(self, i):
        """
        Get differential number i
        """
        if i <= 5:
            d = Whitebear.get_differential(self, i)
        else:
            d = self.mu_disp[i-6, :]
        return d


if __name__ == "__main__":
    # Model testing

    pcs = get_thermopack_model("PC-SAFT")
    pcs.init("C1")
    PCS_functional = pc_saft(1, pcs, T_red=1.1)
    dens_pcs = weighted_densities_pc_saft_1D(1, PCS_functional.R)

    v = pcs.specific_volume(PCS_functional.T,
                            1.0e6,
                            np.array([1.0]),
                            pcs.LIQPH)
    rho = (NA * PCS_functional.d_hs[0] ** 3)/v
    PCS_functional.test_bulk_differentials(rho)

    dens = weighted_densities_pc_saft_1D(1, PCS_functional.R)
    dens.set_testing_values(rho)
    # dens.print(print_utilities=True)
    print("\n")
    PCS_functional.test_differentials(dens)

    # Hard sphere functionals
    # dens = weighted_densities_1D(1, 0.5)
    # dens.set_testing_values()
    # dens.print(print_utilities=True)
    #
    # RF_functional = Rosenfeld(N=1)
    # RF_functional.test_differentials(dens)
    # WB_functional = Whitebear(N=1)
    # WB_functional.test_differentials(dens)
    # WBII_functional = WhitebearMarkII(N=1)
    # WBII_functional.test_differentials(dens)

    # rho = np.array([0.5, 0.1])
    # R = np.array([0.5, 0.3])
    # RF_functional = Rosenfeld(N=1, R=R)
    # RF_functional.test_bulk_differentials(rho)
    # WB_functional = Whitebear(N=1, R=R)
    # WB_functional.test_bulk_differentials(rho)
    # WBII_functional = WhitebearMarkII(N=1, R=R)
    # WBII_functional.test_bulk_differentials(rho)
