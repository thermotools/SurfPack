#!/usr/bin/env python3
import numpy as np
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from constants import NA, RGAS, GRID_UNIT, LenghtUnit
from fmt_functionals import Whitebear
from pyctp import pcsaft

class pc_saft(Whitebear):
    """

    """

    def __init__(self, N, pcs: pcsaft, T_red, phi_disp=1.3862):
        """

        Args:
            pcs (pcsaft): Thermopack object
            T_red (float): Reduced temperature
            R (ndarray): Particle radius for all components
        """
        self.thermo = pcs
        self.T_red = T_red
        self.T = self.T_red * self.thermo.eps_div_kb[0]
        self.d_hs, self.d_T_hs = pcs.hard_sphere_diameters(self.T)

        if GRID_UNIT == LenghtUnit.ANGSTROM:
            self.grid_reducing_lenght = 1.0e-10
        else:
            self.grid_reducing_lenght = self.d_hs[0]
        R = np.zeros(pcs.nc)
        R[:] = 0.5*self.d_hs[:]/self.grid_reducing_lenght
        Whitebear.__init__(self, N, R)
        self.name = "PC-SAFT"
        self.short_name = "PC"
        # Add normalized theta weight
        self.mu_disp = np.zeros((N, pcs.nc))
        self.disp_name = "w_disp"
        self.wf.add_norm_theta_weight(self.disp_name, kernel_radius=2*phi_disp)
        self.diff[self.disp_name] = self.mu_disp

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
            rho_thermo[:] = dens.n[self.disp_name][:, i]
            rho_mix = np.sum(rho_thermo)
            rho_thermo *= 1.0/(NA*self.grid_reducing_lenght**3)
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
        #prdm = dens.n[self.disp_name] > 0.0  # Positive rho_disp value mask
        #print(np.shape(dens.n[self.disp_name]))
        #for i in range(self.nc):
        #    np.logical_and(prdm, dens.n[self.disp_name][i, :] > 0.0, out=prdm)
        #print(np.shape(prdm))
        # prdm = dens.rho_disp > 0.0  # Positive rho_disp value mask
        # for i in range(self.nc):
        #     np.logical_and(prdm, dens.rho_disp_array[:, i] > 0.0, out=prdm)
        # Mask for zero and negative value of rho_disp
        #non_prdm = np.invert(prdm)
        rho_thermo = np.zeros(self.nc)
        V = 1.0
        for i in range(self.n_grid):
        #   if prdm[i]:
            rho_thermo[:] = dens.n[self.disp_name][:, i]
            rho_thermo *= 1.0/(NA*self.grid_reducing_lenght**3)
            a, a_n, = self.thermo.a_dispersion(
                self.T, V, rho_thermo, a_n=True)
            self.mu_disp[i, :] = (a + rho_thermo[:]*a_n[:])
        #self.mu_disp[non_prdm, :] = 0.0

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
        rho_thermo = np.zeros_like(rho_b)
        rho_thermo[:] = rho_b[:]
        rho_thermo *= 1.0/(NA*self.grid_reducing_lenght**3)
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
        rho_thermo = np.zeros_like(rho_b)
        rho_thermo[:] = rho_b[:]
        rho_thermo *= 1.0/(NA*self.grid_reducing_lenght**3)
        rho_mix = np.sum(rho_thermo)
        V = 1.0
        n = rho_thermo
        a, a_n, = self.thermo.a_dispersion(
            self.T, V, n, a_n=True)
        a_n *= n
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
            rho_thermo = np.zeros_like(rho_vec)
            rho_thermo[:] = rho_vec[:]/(NA*self.grid_reducing_lenght**3)
            a, a_n, = self.thermo.a_dispersion(
                self.T, V, rho_thermo, a_n=True)
            phi += rho_mix*a
            dphidn_comb = np.zeros(4 + self.nc)
            dphidn_comb[:4] = dphidn
            dphidn_comb[4:] = a + rho_thermo[:]*a_n[:]
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

    def temperature_differential(self, dens):
        """
        Calculates the functional differentials wrpt. temperature

        Args:
        dens (array_like): weighted densities
        Return:
        np.ndarray: Functional differentials

        """
        d_T = Whitebear.temperature_differential(self, dens)

        rho_thermo = np.zeros(self.nc)
        V = 1.0
        for i in range(self.n_grid):
            rho_thermo[:] = dens.n[self.disp_name][:, i]
            rho_mix = np.sum(rho_thermo)
            rho_thermo *= 1.0/(NA*self.grid_reducing_lenght**3)
            a, a_T, = self.thermo.a_dispersion(
                self.T, V, rho_thermo, a_T=True)
            d_T[i] += rho_mix*a_T

        return d_T

if __name__ == "__main__":
    # Model testing

    pcs = get_thermopack_model("PC-SAFT")
    pcs.init("C1")
    PCS_functional = pc_saft(1, pcs, T_red=110.0/165.0)
    print(PCS_functional.d_hs[0], PCS_functional.T)
    dens_pcs = weighted_densities_pc_saft_1D(1, PCS_functional.R, ms=[1.0])

    v = pcs.specific_volume(PCS_functional.T,
                            1.0e6,
                            np.array([1.0]),
                            pcs.LIQPH)
    rho = (NA * PCS_functional.grid_reducing_lenght ** 3)/v
    PCS_functional.test_bulk_differentials(rho)
    dens = weighted_densities_pc_saft_1D(1, PCS_functional.R, ms=[1.0])
    dens.set_testing_values(rho)
    # dens.print(print_utilities=True)
    PCS_functional.test_differentials(dens)
    corr = PCS_functional.get_bulk_correlation(rho)
    mu = PCS_functional.bulk_excess_chemical_potential(rho)
    print("corr, mu", corr, mu)
