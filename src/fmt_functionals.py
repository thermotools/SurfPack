#!/usr/bin/env python3
import numpy as np
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from weight_functions import WeightFunctions
from constants import LenghtUnit

class bulk_weighted_densities:
    """
    Utility class for calculating bulk states.
    """

    def __init__(self, rho_b, R, ms):
        """

        Args:
            rho_b (ndarray): Bulk densities
            R (ndarray): Particle radius for all components
            ms (ndarray): Segment number for all components
        """
        self.rho_i = np.zeros_like(rho_b)
        self.rho_i[:] = ms*rho_b[:]
        self.n = np.zeros(4)
        self.n[0] = np.sum(self.rho_i)
        self.n[1] = np.sum(R * self.rho_i)
        self.n[2] = 4*np.pi*np.sum(R ** 2 * self.rho_i)
        self.n[3] = 4 * np.pi * np.sum(R ** 3 * self.rho_i) / 3
        self.dndrho = np.zeros((4, np.shape(self.rho_i)[0]))
        self.dndrho[0, :] = ms
        self.dndrho[1, :] = ms*R
        self.dndrho[2, :] = ms*4*np.pi*R**2
        self.dndrho[3, :] = ms*4*np.pi*R**3/3

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

    def __init__(self, N, R=np.array([0.5]), ms=np.array([1.0]), grid_unit=LenghtUnit.ANGSTROM):
        """

        Args:
            N (integer): Grid size
            R (ndarray): Particle radius for all components
            ms (ndarray): Segment number for all components
            grid_unit (LenghtUnit): Information on how lenght is reduced (Deafult: ANGSTROM)
        """
        self.name = "Rosenfeld"
        self.short_name = "RF"
        self.R = R
        self.ms = ms
        self.nc = np.shape(R)[0]
        self.n_grid = N
        # Define units for simulation grid
        self.grid_unit = grid_unit

        # Allocate arrays for differentials
        self.d0 = np.zeros(N)
        self.d1 = np.zeros(N)
        self.d2 = np.zeros(N)
        self.d3 = np.zeros(N)
        self.d1v = np.zeros(N)
        self.d2v = np.zeros(N)
        self.d2eff = np.zeros(N)
        self.d2veff = np.zeros(N)
        self.d_T = np.zeros(N)
        # Set up FMT weights
        self.wf = WeightFunctions()
        self.wf.add_fmt_weights()

        # Differentials
        self.diff = {}
        for wf in self.wf.wfs:
            alias = self.wf.wfs[wf].alias
            if "v" in alias:
                if "1" in alias:
                    self.diff[alias] = self.d1v
                else:
                    self.diff[alias] = self.d2v
            elif "0" in alias:
                self.diff[alias] = self.d0
            elif "1" in alias:
                self.diff[alias] = self.d1
            elif "2" in alias:
                self.diff[alias] = self.d2
            elif "3" in alias:
                self.diff[alias] = self.d3

    def excess_free_energy(self, dens):
        """
        Calculates the excess HS Helmholtz free energy from the weighted densities

        Args:
            dens (array_like): Weighted densities

        Returns:
            array_like: Excess HS Helmholtz free energy ()

        """
        f = np.zeros(dens.n_grid)
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
        bd = bulk_weighted_densities(rho_b, self.R, self.ms)
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
        bd = bulk_weighted_densities(rho_b, self.R, self.ms)
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

    def temperature_differential(self, dens):
        """
        Calculates the functional differentials wrpt. temperature

        Args:
        dens (array_like): weighted densities
        Return:
        np.ndarray: Functional differentials

        """
        self.d_T.fill(0.0)
        return self.d_T

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

    def get_bulk_correlation(self, rho_b, only_hs_system=False):
        """
        Intended only for debugging
        Args:
            rho_b (np.ndarray): Bulk densities
            only_hs_system (bool): Only calculate for hs-system

        Return:
            corr (np.ndarray): One particle correlation function
        """
        bd = bulk_weighted_densities(rho_b, self.R, self.ms)
        _, dphidn = self.bulk_functional_with_differentials(
            bd, only_hs_system=only_hs_system)
        corr = np.zeros(self.nc)
        for i in range(self.nc):
            corr[i] = dphidn[0] + \
                self.R[i] * dphidn[1] + \
                (4 * np.pi * self.R[i] ** 2) * dphidn[2] + \
                4 * np.pi * self.R[i] ** 3 * dphidn[3] / 3
            corr[i] *= self.ms[i]
            if np.shape(dphidn)[0] > 4:
                corr[i] += dphidn[4+i]

        corr = -corr
        return corr

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
            print("Differential: ", i, dFdn_num, self.get_differential(i), (dFdn_num -self.get_differential(i))/self.get_differential(i))

    def test_bulk_differentials(self, rho_b):
        """

        Args:
            rho_b (np.ndarray): Bulk densities

        """
        print("Testing functional " + self.name)
        bd0 = bulk_weighted_densities(rho_b, self.R, self.ms)
        phi, dphidn = self.bulk_functional_with_differentials(bd0)

        print("HS functional differentials:")
        for i in range(4):
            bd = bulk_weighted_densities(rho_b, self.R, self.ms)
            eps = 1.0e-5 * bd.n[i]
            bd.n[i] += eps
            phi2, dphidn2 = self.bulk_functional_with_differentials(bd)
            bd = bulk_weighted_densities(rho_b, self.R, self.ms)
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
            bd = bulk_weighted_densities(rho_b_local, self.R, self.ms)
            phi2, dphidn1 = self.bulk_functional_with_differentials(bd)
            phi2_hs, _ = self.bulk_functional_with_differentials(
                bd, only_hs_system=True)
            rho_b_local[:] = rho_b[:]
            rho_b_local[i] -= eps
            bd = bulk_weighted_densities(rho_b_local, self.R, self.ms)
            phi1, dphidn1 = self.bulk_functional_with_differentials(bd)
            phi1_hs, _ = self.bulk_functional_with_differentials(
                bd, only_hs_system=True)
            dphidrho_num_no_hs = (phi2 - phi2_hs - phi1 + phi1_hs) / (2 * eps)
            dphidrho_num = (phi2 - phi1) / (2 * eps)
            if np.shape(dphidn)[0] > 4:
                print("Differential: ", "[4:]", dphidrho_num_no_hs, np.sum(dphidn[4:]), (dphidrho_num_no_hs - np.sum(dphidn[4:]))/np.sum(dphidn[4:]))
            print("Chemical potential comp.: ", i, dphidrho_num, mu_ex[i])


    def test_eos_differentials(self, V, n):
        """
        Test the functional differentials
        Args:
            V (float): Volume (m3)
            n (np.ndarray): Molar numbers (mol)
        """
        print("No EOS.")


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

    def __init__(self, N, R=np.array([0.5]), ms=np.array([1.0]), grid_unit=LenghtUnit.ANGSTROM):
        """

        Args:
            N (integer): Grid size
            R (ndarray): Particle radius for all components
            ms (ndarray): Segment number for all components
            grid_unit (LenghtUnit): Information on how lenght is reduced (Deafult: ANGSTROM)
        """
        Rosenfeld.__init__(self, N, R, ms, grid_unit=grid_unit)
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
        f = np.zeros(dens.n_grid)
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

    def __init__(self, N, R=np.array([0.5]), ms=np.array([1.0]), grid_unit=LenghtUnit.ANGSTROM):
        """

        Args:
            N (integer): Grid size
            R (ndarray): Radius of particles
            ms (ndarray): Segment number for all components
            grid_unit (LenghtUnit): Information on how lenght is reduced (Deafult: ANGSTROM)
        """
        Whitebear.__init__(self, N, R, ms, grid_unit)
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
            mask = np.full(dens.n_grid, True, dtype=bool)
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
        f = np.zeros(dens.n_grid)
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


if __name__ == "__main__":
    pass
    # Model testing

    # pcs = get_thermopack_model("PC-SAFT")
    # pcs.init("C1")
    # PCS_functional = pc_saft(1, pcs, T_red=110.0/165.0)
    # print(PCS_functional.d_hs[0], PCS_functional.T)
    # dens_pcs = weighted_densities_pc_saft_1D(1, PCS_functional.R, ms=[1.0])

    # v = pcs.specific_volume(PCS_functional.T,
    #                         1.0e6,
    #                         np.array([1.0]),
    #                         pcs.LIQPH)
    # rho = (NA * PCS_functional.d_hs[0] ** 3)/v
    # PCS_functional.test_bulk_differentials(rho)
    # dens = weighted_densities_pc_saft_1D(1, PCS_functional.R, ms=[1.0])
    # dens.set_testing_values(rho)
    # # dens.print(print_utilities=True)
    # PCS_functional.test_differentials(dens)
    # corr = PCS_functional.get_bulk_correlation(rho)
    # mu = PCS_functional.bulk_excess_chemical_potential(rho)
    # print("corr, mu", corr, mu)

    # Hard sphere functionals
    # dens = weighted_densities_1D(1, 0.5)
    # dens.set_testing_values()
    # dens.print(print_utilities=True)
    #
    # RF_functional = Rosenfeld(N=1)
    # corr = RF_functional.get_bulk_correlation(np.array([rho]))
    # mu = RF_functional.bulk_excess_chemical_potential(np.array([rho]))
    # print("corr, mu", corr, mu)

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
