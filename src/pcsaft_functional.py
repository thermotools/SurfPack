#!/usr/bin/env python3
import numpy as np
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from constants import NA, RGAS, LenghtUnit
from fmt_functionals import Whitebear
from pyctp.pcsaft import pcsaft
from pyctp.saft import saft
from weight_functions import WeightFunctionType


class saft_dispersion(Whitebear):
    """

    """

    def __init__(self, N, thermo: saft, T_red, psi_disp=1.3862, grid_unit=LenghtUnit.ANGSTROM):
        """

        Args:
            thermopack (thermo): Thermopack object
            T_red (float): Reduced temperature
            R (ndarray): Particle radius for all components
            psi_disp (float): Width for weighted dispersion density
        """
        self.thermo = thermo
        self.T_red = T_red
        self.T = self.T_red * self.thermo.eps_div_kb[0]
        self.d_hs, self.d_T_hs = thermo.hard_sphere_diameters(self.T)

        if grid_unit == LenghtUnit.ANGSTROM:
            self.grid_reducing_lenght = 1.0e-10
        else:
            self.grid_reducing_lenght = thermo.sigma[0]
        R = np.zeros(thermo.nc)
        R[:] = 0.5*self.d_hs[:]/self.grid_reducing_lenght
        Whitebear.__init__(self, N, R, thermo.m, grid_unit)
        self.name = "Generic-SAFT"
        self.short_name = "GS"
        # Add normalized theta weight
        self.mu_disp = np.zeros((N, thermo.nc))
        self.disp_name = "w_disp"
        self.wf.add_norm_theta_weight(self.disp_name, kernel_radius=2*psi_disp)
        self.diff[self.disp_name] = self.mu_disp
        # Add storage container for differentials only depending on local density
        # No convolution required
        self.mu_of_rho = np.zeros((N, self.nc))

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
        self.mu_of_rho.fill(0.0)
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
        # Dispersion contributions
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
        Temperature dependence through weigthed densities calculated elewhere
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
        print("Dispersion functional:")
        a, a_t, a_v, a_n, a_tt, a_vv, a_tv, a_tn, a_vn, a_nn = self.thermo.a_dispersion(
            self.T, V, n, a_t=True, a_v=True, a_n=True, a_tt=True, a_vv=True,
            a_tv=True, a_tn=True, a_vn=True, a_nn=True)

        eps = 1.0e-5
        dT = self.T*eps
        ap, ap_t, ap_v, ap_n = self.thermo.a_dispersion(self.T + dT, V, n, a_t=True, a_v=True, a_n=True)
        am, am_t, am_v, am_n = self.thermo.a_dispersion(self.T - dT, V, n, a_t=True, a_v=True, a_n=True)
        print(f"a_T: {a_t}, {(ap-am)/2/dT}")
        print(f"a_TT: {a_tt}, {(ap_t-am_t)/2/dT}")
        print(f"a_TV: {a_tv}, {(ap_v-am_v)/2/dT}")
        print(f"a_Tn: {a_tn}, {(ap_n-am_n)/2/dT}")
        dV = V*eps
        ap, ap_t, ap_v, ap_n = self.thermo.a_dispersion(self.T, V + dV, n, a_t=True, a_v=True, a_n=True)
        am, am_t, am_v, am_n = self.thermo.a_dispersion(self.T, V - dV, n, a_t=True, a_v=True, a_n=True)
        print(f"a_V: {a_v}, {(ap-am)/2/dV}")
        print(f"a_VV: {a_vv}, {(ap_v-am_v)/2/dV}")
        print(f"a_TV: {a_tv}, {(ap_t-am_t)/2/dV}")
        print(f"a_Vn: {a_vn}, {(ap_n-am_n)/2/dV}")
        eps = 1.0e-5
        dn = np.zeros_like(n)
        dn[0] = n[0]*eps
        ap, ap_t, ap_v, ap_n = self.thermo.a_dispersion(self.T, V, n + dn, a_t=True, a_v=True, a_n=True)
        am, am_t, am_v, am_n = self.thermo.a_dispersion(self.T, V, n - dn, a_t=True, a_v=True, a_n=True)
        print(f"a_n: {a_n}, {(ap-am)/2/dn[0]}")
        print(f"a_Vn: {a_vn}, {(ap_v-am_v)/2/dn[0]}")
        print(f"a_Tn: {a_tn}, {(ap_t-am_t)/2/dn[0]}")
        print(f"a_nn: {a_nn}, {(ap_n-am_n)/2/dn[0]}")

class pc_saft(saft_dispersion):
    """

    """

    def __init__(self,
                 N,
                 pcs: pcsaft,
                 T_red,
                 psi_disp=1.3862,
                 psi_rho_hc=1.0,
                 psi_lambda_hc=1.0,
                 grid_unit=LenghtUnit.ANGSTROM):
        """

        Args:
            N (int): Number of grid points
            pcs (pcsaft): Thermopack object
            T_red (float): Reduced temperature
            psi_disp (float): Width for weighted dispersion density
            psi_rho_hc (float): Width for weighted hard-chain density rho_hc
            psi_lambda_hc (float): Width for weighted hard-chain density lambda_hc
        """
        saft_dispersion.__init__(self,
                                 N,
                                 pcs,
                                 T_red,
                                 psi_disp=psi_disp,
                                 grid_unit=grid_unit)
        self.name = "PC-SAFT"
        self.short_name = "PC"
        self.chain_functional_active = np.full((pcs.nc), False, dtype=bool)
        for i in range(pcs.nc):
            if abs(pcs.m[i] - 1.0) > 1.0e-8:
                self.chain_functional_active[i] = True
        if np.any(self.chain_functional_active):
            # Add normalized theta weight
            self.mu_rho_hc = np.zeros((N, pcs.nc))
            self.rho_hc_name = "w_rho_hc"
            self.wf.add_norm_theta_weight(self.rho_hc_name, kernel_radius=2*psi_rho_hc)
            self.diff[self.rho_hc_name] = self.mu_rho_hc
            # Add dirac delta weight reducing to rho in bulk
            self.mu_lambda_hc = np.zeros((N, pcs.nc))
            self.lambda_hc_name = "w_lambda_hc"
            self.wf.add_weight(alias=self.lambda_hc_name,
                               kernel_radius=2*psi_lambda_hc,
                               wf_type=WeightFunctionType.DELTA,
                               prefactor="1.0/(4.0*pi*(R*Psi)**2)")
            self.diff[self.lambda_hc_name] = self.mu_lambda_hc

    def excess_free_energy(self, dens):
        """
        Calculates the excess HS Helmholtz free energy from the weighted densities

        Args:
        dens (array_like): Weighted densities

        Returns:
        array_like: Excess HS Helmholtz free energy ()

        """
        f = saft_dispersion.excess_free_energy(self, dens)
        if np.any(self.chain_functional_active):
            rho_hc_thermo = np.zeros(self.nc)
            lambda_hc = np.zeros(self.nc)
            V = 1.0
            for i in range(self.n_grid):
                rho_hc_thermo[:] = dens.n[self.rho_hc_name][:, i]
                lambda_hc[:] = dens.n[self.lambda_hc_name][:, i]
                rho_hc_thermo *= 1.0/(NA*self.grid_reducing_lenght**3)
                for j in range(self.nc):
                    if self.chain_functional_active[j]:
                        rho_j = dens.rho.densities[j][i]
                        lng_jj, = self.thermo.lng_ii(self.T, volume=V, n=rho_hc_thermo, i=j+1)
                        f_chain = rho_j*(np.log(rho_j) - 1.0)
                        f_chain -= rho_j*(lng_jj + np.log(lambda_hc[j]) - 1.0)
                        f[i] += (self.thermo.m[j]-1.0)*f_chain

        return f

    def differentials(self, dens):
        """
        Calculates the functional differentials wrpt. the weighted densities

        Args:
        dens (array_like): weighted densities
        diff (array_like): Functional differentials

        """
        saft_dispersion.differentials(self, dens)
        if np.any(self.chain_functional_active):
            rho_hc_thermo = np.zeros(self.nc)
            lambda_hc = np.zeros(self.nc)
            V = 1.0
            for i in range(self.n_grid):
                rho_hc_thermo[:] = dens.n[self.rho_hc_name][:, i]
                lambda_hc[:] = dens.n[self.lambda_hc_name][:, i]
                rho_hc_thermo *= 1.0/(NA*self.grid_reducing_lenght**3)
                for j in range(self.nc):
                    if self.chain_functional_active[j]:
                        rho_j = dens.rho.densities[j][i]
                        lng_jj, lng_jj_n = self.thermo.lng_ii(self.T, volume=V, n=rho_hc_thermo, i=j+1, lng_n=True)
                        lng_jj_n /= (NA*self.grid_reducing_lenght**3) # Reducing unit
                        # Contribution not to be convolved:
                        self.mu_of_rho[i, j] += (self.thermo.m[j]-1.0)*(np.log(rho_j) - lng_jj - np.log(lambda_hc[j]) + 1.0)
                        # Convolved contributions:
                        self.mu_rho_hc[i, j] = -(self.thermo.m[j]-1.0)*rho_j*lng_jj_n
                        self.mu_lambda_hc[i, j] = -(self.thermo.m[j]-1.0)*rho_j/lambda_hc[j]

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
        if np.any(self.chain_functional_active):
            # PC-SAFT contributions
            rho_thermo = np.zeros_like(rho_b)
            rho_thermo[:] = rho_b[:]
            rho_thermo *= 1.0/(NA*self.grid_reducing_lenght**3)
            rho_mix = np.sum(rho_thermo)
            V = 1.0/rho_mix
            n = rho_thermo/rho_mix
            for j in range(self.nc):
                if self.chain_functional_active[j]:
                    lng_jj, lng_jj_V = self.thermo.lng_ii(self.T, volume=V, n=n, i=j+1, lng_V=True)
                    a_V = -(self.thermo.m[j]-1.0)*rho_b[j]*(lng_jj_V - lng_jj/V)
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
        if np.any(self.chain_functional_active):
            # Hard-chain contributions
            rho_thermo = np.zeros_like(rho_b)
            rho_thermo[:] = rho_b[:]
            rho_thermo *= 1.0/(NA*self.grid_reducing_lenght**3)
            rho_mix = np.sum(rho_thermo)
            V = 1.0
            n = rho_thermo
            for j in range(self.nc):
                if self.chain_functional_active[j]:
                    lng_jj, lng_jj_n = self.thermo.lng_ii(self.T, volume=V, n=n, i=j+1, lng_n=True)
                    a_n = -(self.thermo.m[j]-1.0)*(rho_thermo[j]*lng_jj_n + lng_jj)
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
        phi, dphidn = saft_dispersion.bulk_functional_with_differentials(self, bd, only_hs_system)
        if not only_hs_system and np.any(self.chain_functional_active):
            rho_vec = bd.rho_i
            rho_mix = np.sum(rho_vec)
            V = 1.0
            rho_thermo = np.zeros_like(rho_vec)
            rho_thermo[:] = rho_vec[:]/(NA*self.grid_reducing_lenght**3)
            n = np.shape(dphidn)[0]
            dphidn_comb = np.zeros(n + self.nc)
            dphidn_comb[:n] = dphidn
            for j in range(self.nc):
                if self.chain_functional_active[j]:
                    lng_jj, lng_jj_n = self.thermo.lng_ii(self.T, volume=V, n=rho_thermo, i=j+1, lng_n=True)
                    phi -= (self.thermo.m[j]-1.0)*rho_b[j]*lng_jj
                    a_n = -(self.thermo.m[j]-1.0)*(rho_thermo[j]*lng_jj_n + lng_jj/V)
                    dphidn_comb[n:] += a_n

        else:
            dphidn_comb = dphidn
        return phi, dphidn_comb

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
        if np.any(self.chain_functional_active):
            rho_hc_thermo = np.zeros(self.nc)
            V = 1.0
            for i in range(self.n_grid):
                rho_hc_thermo[:] = dens.n[self.rho_hc_name][:, i]
                rho_hc_thermo *= 1.0/(NA*self.grid_reducing_lenght**3)
                for j in range(self.nc):
                    if self.chain_functional_active[j]:
                        rho_j = dens.rho.densities[j][i]
                        lng_jj, lng_jj_t, = self.thermo.lng_ii(self.T, volume=V, n=rho_hc_thermo, i=j+1, lng_t=True)
                        d_T[i] += -(self.thermo.m[j]-1.0)*rho_j*lng_jj_t
        return d_T


    def test_eos_differentials(self, V, n):
        """
        Test the functional differentials
        Args:
            V (float): Volume (m3)
            n (np.ndarray): Molar numbers (mol)
        """
        saft_dispersion.test_eos_differentials(self, V, n)
        if np.any(self.chain_functional_active):
            print("Chain functional:")
            lng, lng_t, lng_v, lng_n, lng_tt, lng_tv, lng_vv, lng_tn, lng_vn, lng_nn = self.thermo.lng_ii(
                self.T, V, n, 1, lng_t=True, lng_v=True, lng_n=True, lng_tt=True, lng_vv=True,
                lng_tv=True, lng_tn=True, lng_vn=True, lng_nn=True)
            print("lng",lng)
            eps = 1.0e-5
            dT = self.T*eps
            lngp, lngp_t, lngp_v, lngp_n = self.thermo.lng_ii(self.T + dT, V, n, 1, lng_t=True, lng_v=True, lng_n=True)
            lngm, lngm_t, lngm_v, lngm_n = self.thermo.lng_ii(self.T - dT, V, n, 1, lng_t=True, lng_v=True, lng_n=True)
            print(f"lng_T: {lng_t}, {(lngp-lngm)/2/dT}")
            print(f"lng_TT: {lng_tt}, {(lngp_t-lngm_t)/2/dT}")
            print(f"lng_TV: {lng_tv}, {(lngp_v-lngm_v)/2/dT}")
            print(f"lng_Tn: {lng_tn}, {(lngp_n-lngm_n)/2/dT}")
            dV = V*eps
            lngp, lngp_t, lngp_v, lngp_n = self.thermo.lng_ii(self.T, V + dV, n, 1, lng_t=True, lng_v=True, lng_n=True)
            lngm, lngm_t, lngm_v, lngm_n = self.thermo.lng_ii(self.T, V - dV, n, 1, lng_t=True, lng_v=True, lng_n=True)
            print(f"lng_V: {lng_v}, {(lngp-lngm)/2/dV}")
            print(f"lng_VV: {lng_vv}, {(lngp_v-lngm_v)/2/dV}")
            print(f"lng_TV: {lng_tv}, {(lngp_t-lngm_t)/2/dV}")
            print(f"lng_Vn: {lng_vn}, {(lngp_n-lngm_n)/2/dV}")
            eps = 1.0e-5
            dn = np.zeros_like(n)
            dn[0] = n[0]*eps
            lngp, lngp_t, lngp_v, lngp_n = self.thermo.lng_ii(self.T, V, n + dn, 1, lng_t=True, lng_v=True, lng_n=True)
            lngm, lngm_t, lngm_v, lngm_n = self.thermo.lng_ii(self.T, V, n - dn, 1, lng_t=True, lng_v=True, lng_n=True)
            print(f"lng_n: {lng_n}, {(lngp-lngm)/2/dn[0]}")
            print(f"lng_Vn: {lng_vn}, {(lngp_v-lngm_v)/2/dn[0]}")
            print(f"lng_Tn: {lng_tn}, {(lngp_t-lngm_t)/2/dn[0]}")
            print(f"lng_nn: {lng_nn}, {(lngp_n-lngm_n)/2/dn[0]}")


if __name__ == "__main__":
    pass
