#!/usr/bin/env python3
import numpy as np
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from constants import NA, RGAS, LenghtUnit
from pcsaft_functional import saft_dispersion
from pyctp.ljs_bh import ljs_bh
from pyctp.ljs_wca import ljs_wca_base, ljs_wca, ljs_uv

class ljs_bh_functional(saft_dispersion):
    """

    """

    def __init__(self, N, ljs: ljs_bh, T_red, psi_disp=1.0002, grid_unit=LenghtUnit.ANGSTROM):
        """

        Args:
            N (int): Size of grid
            ljs (ljs_bh): Thermopack object
            T_red (float): Reduced temperature
            psi_disp (float): Width for weighted dispersion density
            grid_unit (LenghtUnit): Unit used for grid
        """
        saft_dispersion.__init__(self,
                                 N,
                                 ljs,
                                 T_red,
                                 psi_disp=psi_disp,
                                 grid_unit=grid_unit)
        self.name += "Lennard-Jones-Spline-BH"
        self.short_name = "LJS-BH"

class ljs_wca_base_functional(saft_dispersion):
    """

    """

    def __init__(self, N, ljs: ljs_wca_base, T_red, psi_disp=1.0002, psi_soft_rep=1.0002, grid_unit=LenghtUnit.ANGSTROM):
        """

        Args:
            N (int): Size of grid
            ljs (ljs_wca_base): Thermopack object
            T_red (float): Reduced temperature
            psi_disp (float): Width for weighted dispersion density
            psi_soft_rep (float): Width for weighted soft-repulsion density
            grid_unit (LenghtUnit): Unit used for grid
        """
        saft_dispersion.__init__(self, N,
                                 ljs,
                                 T_red,
                                 psi_disp=psi_disp,
                                 grid_unit=grid_unit)
        self.name += "Lennard-Jones-Spline-WCA"
        self.short_name = "LJS-WCA"
        # Add normalized theta weight
        self.mu_soft_rep = np.zeros((N, ljs.nc))
        self.soft_rep_name = "w_soft_rep"
        self.wf.add_norm_theta_weight(self.soft_rep_name, kernel_radius=2*psi_soft_rep)
        self.diff[self.soft_rep_name] = self.mu_soft_rep


    def excess_free_energy(self, dens):
        """
        Calculates the excess HS Helmholtz free energy from the weighted densities

        Args:
        dens (array_like): Weighted densities

        Returns:
        array_like: Excess HS Helmholtz free energy ()

        """
        f = saft_dispersion.excess_free_energy(self, dens)
        rho_thermo = np.zeros(self.nc)
        V = 1.0
        for i in range(len(f)):
            rho_thermo[:] = dens.n[self.soft_rep_name][:, i]
            rho_mix = np.sum(rho_thermo)
            rho_thermo *= 1.0/(NA*self.grid_reducing_lenght**3)
            a, = self.thermo.a_soft_repulsion(self.T, V, rho_thermo)
            f[i] += rho_mix*a

        return f

    def differentials(self, dens):
        """
        Calculates the functional differentials wrpt. the weighted densities

        Args:
        dens (array_like): weighted densities
        diff (array_like): Functional differentials

        """
        saft_dispersion.differentials(self, dens)

        # All densities must be positive
        #prdm = dens.n[self.soft_rep_name] > 0.0  # Positive rho_disp value mask
        #print(np.shape(dens.n[self.soft_rep_name]))
        #for i in range(self.nc):
        #    np.logical_and(prdm, dens.n[self.soft_rep_name][i, :] > 0.0, out=prdm)
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
            rho_thermo[:] = dens.n[self.soft_rep_name][:, i]
            rho_thermo *= 1.0/(NA*self.grid_reducing_lenght**3)
            a, a_n, = self.thermo.a_soft_repulsion(
                self.T, V, rho_thermo, a_n=True)
            self.mu_soft_rep[i, :] = (a + rho_thermo[:]*a_n[:])
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
        z = saft_dispersion.bulk_compressibility(self, rho_b)
        # PC-SAFT contributions
        rho_thermo = np.zeros_like(rho_b)
        rho_thermo[:] = rho_b[:]
        rho_thermo *= 1.0/(NA*self.grid_reducing_lenght**3)
        rho_mix = np.sum(rho_thermo)
        V = 1.0/rho_mix
        n = rho_thermo/rho_mix
        a, a_V, = self.thermo.a_soft_repulsion(
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
        mu_ex = saft_dispersion.bulk_excess_chemical_potential(self, rho_b)
        # PC-SAFT contributions
        rho_thermo = np.zeros_like(rho_b)
        rho_thermo[:] = rho_b[:]
        rho_thermo *= 1.0/(NA*self.grid_reducing_lenght**3)
        rho_mix = np.sum(rho_thermo)
        V = 1.0
        n = rho_thermo
        a, a_n, = self.thermo.a_soft_repulsion(
            self.T, V, n, a_n=True)
        a_n *= n
        a_n += a
        mu_ex += a_n
        return mu_ex

    def temperature_differential(self, dens):
        """
        Calculates the functional differentials wrpt. temperature
        Temperature dependence through weigthed densities calculated elewhere
        Args:
        dens (array_like): weighted densities
        Return:
        np.ndarray: Functional differentials

        """
        d_T = saft_dispersion.temperature_differential(self, dens)

        rho_thermo = np.zeros(self.nc)
        V = 1.0
        for i in range(self.n_grid):
            rho_thermo[:] = dens.n[self.soft_rep_name][:, i]
            rho_mix = np.sum(rho_thermo)
            rho_thermo *= 1.0/(NA*self.grid_reducing_lenght**3)
            a, a_T, = self.thermo.a_soft_repulsion(
                self.T, V, rho_thermo, a_t=True)
            d_T[i] += rho_mix*a_T

        return d_T

    def test_eos_differentials(self, V, n):
        """
        Test the functional differentials
        Args:
            V (float): Volume (m3)
            n (np.ndarray): Molar numbers (mol)
        """
        saft_dispersion.test_eos_differentials(self, V, n)
        print("Soft repulsion functional:")
        a, a_t, a_v, a_n, a_tt, a_vv, a_tv, a_tn, a_vn, a_nn = self.thermo.a_soft_repulsion(
            self.T, V, n, a_t=True, a_v=True, a_n=True, a_tt=True, a_vv=True,
            a_tv=True, a_tn=True, a_vn=True, a_nn=True)

        eps = 1.0e-5
        dT = self.T*eps
        ap, ap_t, ap_v, ap_n = self.thermo.a_soft_repulsion(self.T + dT, V, n, a_t=True, a_v=True, a_n=True)
        am, am_t, am_v, am_n = self.thermo.a_soft_repulsion(self.T - dT, V, n, a_t=True, a_v=True, a_n=True)
        print(f"a_T: {a_t}, {(ap-am)/2/dT}")
        print(f"a_TT: {a_tt}, {(ap_t-am_t)/2/dT}")
        print(f"a_TV: {a_tv}, {(ap_v-am_v)/2/dT}")
        print(f"a_Tn: {a_tn}, {(ap_n-am_n)/2/dT}")
        dV = V*eps
        ap, ap_t, ap_v, ap_n = self.thermo.a_soft_repulsion(self.T, V + dV, n, a_t=True, a_v=True, a_n=True)
        am, am_t, am_v, am_n = self.thermo.a_soft_repulsion(self.T, V - dV, n, a_t=True, a_v=True, a_n=True)
        print(f"a_V: {a_v}, {(ap-am)/2/dV}")
        print(f"a_VV: {a_vv}, {(ap_v-am_v)/2/dV}")
        print(f"a_TV: {a_tv}, {(ap_t-am_t)/2/dV}")
        print(f"a_Vn: {a_vn}, {(ap_n-am_n)/2/dV}")
        eps = 1.0e-5
        dn = np.zeros_like(n)
        dn[0] = n[0]*eps
        ap, ap_t, ap_v, ap_n = self.thermo.a_soft_repulsion(self.T, V, n + dn, a_t=True, a_v=True, a_n=True)
        am, am_t, am_v, am_n = self.thermo.a_soft_repulsion(self.T, V, n - dn, a_t=True, a_v=True, a_n=True)
        print(f"a_n: {a_n}, {(ap-am)/2/dn[0]}")
        print(f"a_Vn: {a_vn}, {(ap_v-am_v)/2/dn[0]}")
        print(f"a_Tn: {a_tn}, {(ap_t-am_t)/2/dn[0]}")
        print(f"a_nn: {a_nn}, {(ap_n-am_n)/2/dn[0]}")


class ljs_uv_functional(ljs_wca_base_functional):
    """

    """

    def __init__(self, N, ljs: ljs_uv, T_red, psi_disp=1.0002, psi_soft_rep=1.0002, grid_unit=LenghtUnit.ANGSTROM):
        """

        Args:
            N (int): Size of grid
            ljs (ljs_uv): Thermopack object
            T_red (float): Reduced temperature
            psi_disp (float): Width for weighted dispersion density
            psi_soft_rep (float): Width for weighted soft-repulsion density
            grid_unit (LenghtUnit): Unit used for grid
        """
        ljs_wca_base_functional.__init__(self,
                                         N,
                                         ljs,
                                         T_red,
                                         psi_disp=psi_disp,
                                         psi_soft_rep=psi_soft_rep,
                                         grid_unit=grid_unit)
        self.name += "Lennard-Jones-Spline-UV"
        self.short_name = "LJS-UV"

class ljs_wca_functional(ljs_wca_base_functional):
    """

    """

    def __init__(self, N, ljs: ljs_wca, T_red, psi_disp=1.0002, psi_soft_rep=1.0002, grid_unit=LenghtUnit.ANGSTROM):
        """

        Args:
            N (int): Size of grid
            ljs (ljs_wca): Thermopack object
            T_red (float): Reduced temperature
            psi_disp (float): Width for weighted dispersion density
            psi_soft_rep (float): Width for weighted soft-repulsion density
            grid_unit (LenghtUnit): Unit used for grid
        """
        ljs_wca_base_functional.__init__(self,
                                         N,
                                         ljs,
                                         T_red,
                                         psi_disp=psi_disp,
                                         psi_soft_rep=psi_soft_rep,
                                         grid_unit=grid_unit)
        self.name += "Lennard-Jones-Spline-WCA"
        self.short_name = "LJS-WCA"


if __name__ == "__main__":
    # Model testing
    pass
