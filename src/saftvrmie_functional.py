#!/usr/bin/env python3
import numpy as np
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from constants import NA, RGAS, LenghtUnit
from pcsaft_functional import saft_dispersion
from pyctp.saftvrmie import saftvrmie
from pyctp.saftvrqmie import saftvrqmie

class saftvrqmie_functional(saft_dispersion):
    """

    """

    def __init__(self, N, svrqm: saftvrqmie, T_red, psi_disp=1.3862, grid_unit=LenghtUnit.ANGSTROM):
        """

        Args:
            N (int): Size of grid
            svrqm (saftvrqmie): Thermopack object
            T_red (float): Reduced temperature
            psi_disp (float): Width for weighted dispersion density
            grid_unit (LenghtUnit): Unit used for grid
        """
        saft_dispersion.__init__(self,
                                 N,
                                 svrqm,
                                 T_red,
                                 psi_disp=psi_disp,
                                 grid_unit=grid_unit)
        self.name = "SAFTVRQ-MIE"
        self.short_name = "SVRQM"
        # Set up non-additive correction
        self.na_enabled = False
        self.delta_ij = None
        self.delta_T_ij = None
        self.d_ij = None
        self.d_T_ij = None
        self.mu_ij = None
        self.mu_ij_T = None
        if svrqm.nc > 1:
            _, self.na_enabled = svrqm.test_fmt_compatibility()
        if self.na_enabled:
            self.d_ij = np.zeors((svrqm.nc, svrqm.nc))
            self.delta_ij = np.zeors((svrqm.nc, svrqm.nc))
            self.d_T_ij = np.zeors((svrqm.nc, svrqm.nc))
            self.delta_T_ij = np.zeors((svrqm.nc, svrqm.nc))
            self.mu_ij = np.zeors((svrqm.nc, svrqm.nc))
            self.mu_ij_T = np.zeors((svrqm.nc, svrqm.nc))
            for i in range(svrqm.nc):
                self.d_ij[i,i] = self.d_hs[i]
                self.d_T_ij[i,i] = self.d_T_hs[i]
                self.delta_ij[i,i] = self.d_ij[i,i]
                self.delta_T_ij[i,i] = self.d_T_ij[i,i]
                self.mu_ij[i,i] = 0.5*self.d_hs[i]
                self.mu_ij_T[i,i] = 0.5*self.d_T_hs[i]
                for j in range(i+1,svrqm.nc):
                    self.d_ij[i,j] = 0.5*(self.d_hs[i]+self.d_hs[j])
                    self.d_ij[j,i] = self.d_ij[i,j]
                    self.d_T_ij[i,j] = 0.5*(self.d_T_hs[i]+self.d_T_hs[j])
                    self.d_T_ij[j,i] = self.d_T_ij[i,j]
                    self.delta_ij[i,j], self.delta_T_ij[i,j] = svrqm.hard_sphere_diameters_ij(i+1, j+1, self.T)
                    self.delta_ij[j,i] = self.delta_ij[i,j]
                    self.delta_T_ij[j,i] = self.delta_T_ij[i,j]
                    self.mu_ij[i,j] = self.d_hs[i]*self.d_hs[j]/(self.d_hs[i]+self.d_hs[j])
                    self.mu_ij_T[i,j] = (self.d_T_hs[i]*self.d_hs[j] + self.d_hs[i]*self.d_T_hs[j])/(self.d_hs[i]+self.d_hs[j]) \
                        - (self.d_T_hs[i] + self.d_T_hs[j] )*self.mu_ij[i,j]/(self.d_hs[i]+self.d_hs[j])

    def excess_free_energy(self, dens):
        """
        Calculates the excess HS Helmholtz free energy from the weighted densities

        Args:
        dens (array_like): Weighted densities

        Returns:
        array_like: Excess HS Helmholtz free energy ()

        """
        f = saft_dispersion.excess_free_energy(self, dens)
        if self.na_enabled:
            for i in range(self.n_grid):
                n_alpha = dens.get_fmt_densities(i)
                for j in range(self.thermo.nc):
                    n_alpha_j = dens.comp_weighted_densities[j].get_fmt_densities(i)
                    for k in range(j+1,self.thermo.nc):
                        n_alpha_k = dens.comp_weighted_densities[k].get_fmt_densities(i)
                        g_jk, = self.thermo.calc_bmcsl_gij_fmt(n_alpha, self.mu_ij[j,k])
                        f[i] -= 4*np.pi*n_alpha_j[0]*n_alpha_k[0]*self.d_ij[j,k]**2*g_jk*(self.d_ij[j,k] - self.delta_ij[j,k])

        return f

    def differentials(self, dens):
        """
        Calculates the functional differentials wrpt. the weighted densities

        Args:
        dens (array_like): weighted densities
        diff (array_like): Functional differentials

        """
        saft_dispersion.differentials(self, dens)
        if self.na_enabled:
            for i in range(self.n_grid):
                n_alpha = dens.get_fmt_densities(i)
                for j in range(self.thermo.nc):
                    n_alpha_j = dens.comp_weighted_densities[j].get_fmt_densities(i)
                    for k in range(j+1,self.thermo.nc):
                        n_alpha_k = dens.comp_weighted_densities[k].get_fmt_densities(i)
                        g_jk, g_jk_n, = self.thermo.calc_bmcsl_gij_fmt(n_alpha, self.mu_ij[j,k], g_ij_n=True)
                        ck = -4*np.pi*self.d_ij[j,k]**2*(self.d_ij[j,k] - self.delta_ij[j,k])
                        self.d0[i, j] += ck*n_alpha_k[0]*g_jk
                        self.d0[i, k] += ck*n_alpha_j[0]*g_jk
                        ck *= n_alpha_j[0]*n_alpha_k[0]
                        self.d0[i, :] += ck*g_jk_n[0]
                        self.d1[i, :] += ck*g_jk_n[1]
                        self.d2[i, :] += ck*g_jk_n[2]
                        self.d3[i, :] += ck*g_jk_n[3]
                        self.d1v[i, :] += ck*g_jk_n[4]
                        self.d2v[i, :] += ck*g_jk_n[5]

    def bulk_compressibility(self, rho_b):
        """
        Add contribution of non-additive functional to compressibility.
        Multiply by rho*kB*T to get pressure.

        Args:
            rho_b (ndarray): Bulk densities

        Returns:
            float: compressibility
        """
        z = saft_dispersion.bulk_compressibility(self, rho_b)
        if self.na_enabled:
            bd = bulk_weighted_densities(rho_b, self.R, self.ms)
            phi, dphidn, d0_i = self.bulk_na_functional_with_differentials(bd)
            beta_p_ex = - phi + np.sum(dphidn * bd.n) + np.sum(d0_i * bd.na[0,:])
            z_r = beta_p_ex/bd.n[0]
            z += z_r
        return z

    def bulk_excess_chemical_potential(self, rho_b):
        """
        Calculates the reduced excess chemical potential.

        Args:
        rho_b (ndarray): Bulk densities

        Returns:
        float: Excess reduced chemical potential ()

        """
        mu_ex = saft_dispersion.bulk_excess_chemical_potential(self, rho_b)
        if self.na_enabled:
            bd = bulk_weighted_densities(rho_b, self.R, self.ms)
            phi, dphidn, d0_i = self.bulk_na_functional_with_differentials(bd)
            for i in range(self.nc):
                mu_ex[i] += np.sum(dphidn[:4] * bd.dndrho[:, i]) + np.sum(d0_i * bd.dndrho[:, i])
        return mu_ex

    def bulk_na_functional_with_differentials(self, bd):
        """
        Calculates the functional differentials wrpt. the weighted densities
        in the bulk phase.

        Args:
        bd (bulk_weighted_densities): bulk_weighted_densities
        only_hs_system (bool): Only calculate for hs-system
        """

        phi = 0.0
        d0 = 0.0
        d1 = 0.0
        d2 = 0.0
        d3 = 0.0
        d0_i = np.zeros_like(bd.na)
        if self.na_enabled:
            n_alpha = np.zeros(bd.n[:]+[0.0, 0.0])
            for j in range(self.thermo.nc):
                for k in range(j+1,self.thermo.nc):
                    g_jk, g_jk_n, = self.thermo.calc_bmcsl_gij_fmt(n_alpha, self.mu_ij[j,k], g_ij_n=True)
                    ck = -4*np.pi*self.d_ij[j,k]**2*(self.d_ij[j,k] - self.delta_ij[j,k])
                    phi += ck*bd.na[0,j]*bd.na[0,k]*g_jk
                    d0_i[j] += ck*bd.na[0,k]*g_jk
                    d0_i[k] += ck*bd.na[0,j]*g_jk
                    ck *= bd.na[0,j]*bd.na[0,k]
                    d0 += ck*g_jk_n[0]
                    d1 += ck*g_jk_n[1]
                    d2 += ck*g_jk_n[2]
                    d3 += ck*g_jk_n[3]
        dphidn = np.array([d0, d1, d2, d3])
        return phi, dphidn, d0_i

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
        if self.na_enabled:
            for i in range(self.n_grid):
                n_alpha = dens.get_fmt_densities(i)
                for j in range(self.thermo.nc):
                    n_alpha_j = dens.comp_weighted_densities[j].get_fmt_densities(i)
                    for k in range(j+1,self.thermo.nc):
                        n_alpha_k = dens.comp_weighted_densities[k].get_fmt_densities(i)
                        g_jk, g_jk_T, = self.thermo.calc_bmcsl_gij_fmt(n_alpha, self.mu_ij[j,k], mu_ij_T=self.mu_ij_T[j,k])
                        ck = -4*np.pi*n_alpha_j[0]*n_alpha_k[0]
                        g_jk*self.d_ij[j,k]**2*(self.d_ij[j,k] - self.delta_ij[j,k])
                        d_T[i] += ck*g_jk_T*self.d_ij[j,k]**2*(self.d_ij[j,k] - self.delta_ij[j,k]) \
                            + 2*ck*g_jk*self.d_ij[j,k]*(self.d_ij[j,k] - self.delta_ij[j,k])*self.d_T_ij[j,k] \
                            + ck*g_jk*self.d_ij[j,k]**2*self.d_T_ij[j,k] \
                            - ck*g_jk*self.d_ij[j,k]**2*self.delta_T_ij[j,k]
        return d_T


    def test_eos_differentials(self, V, n):
        """
        Test the functional differentials
        Args:
            V (float): Volume (m3)
            n (np.ndarray): Molar numbers (mol)
        """
        saft_dispersion.test_eos_differentials(self, V, n)
        if self.na_enabled:
            print("Non-additive functional:")
            # lng, lng_t, lng_v, lng_n, lng_tt, lng_tv, lng_vv, lng_tn, lng_vn, lng_nn = self.thermo.lng_ii(
            #     self.T, V, n, 1, lng_t=True, lng_v=True, lng_n=True, lng_tt=True, lng_vv=True,
            #     lng_tv=True, lng_tn=True, lng_vn=True, lng_nn=True)
            # print("lng",lng)
            # eps = 1.0e-5
            # dT = self.T*eps
            # lngp, lngp_t, lngp_v, lngp_n = self.thermo.lng_ii(self.T + dT, V, n, 1, lng_t=True, lng_v=True, lng_n=True)
            # lngm, lngm_t, lngm_v, lngm_n = self.thermo.lng_ii(self.T - dT, V, n, 1, lng_t=True, lng_v=True, lng_n=True)
            # print(f"lng_T: {lng_t}, {(lngp-lngm)/2/dT}")
            # print(f"lng_TT: {lng_tt}, {(lngp_t-lngm_t)/2/dT}")
            # print(f"lng_TV: {lng_tv}, {(lngp_v-lngm_v)/2/dT}")
            # print(f"lng_Tn: {lng_tn}, {(lngp_n-lngm_n)/2/dT}")
            # dV = V*eps
            # lngp, lngp_t, lngp_v, lngp_n = self.thermo.lng_ii(self.T, V + dV, n, 1, lng_t=True, lng_v=True, lng_n=True)
            # lngm, lngm_t, lngm_v, lngm_n = self.thermo.lng_ii(self.T, V - dV, n, 1, lng_t=True, lng_v=True, lng_n=True)
            # print(f"lng_V: {lng_v}, {(lngp-lngm)/2/dV}")
            # print(f"lng_VV: {lng_vv}, {(lngp_v-lngm_v)/2/dV}")
            # print(f"lng_TV: {lng_tv}, {(lngp_t-lngm_t)/2/dV}")
            # print(f"lng_Vn: {lng_vn}, {(lngp_n-lngm_n)/2/dV}")
            # eps = 1.0e-5
            # dn = np.zeros_like(n)
            # dn[0] = n[0]*eps
            # lngp, lngp_t, lngp_v, lngp_n = self.thermo.lng_ii(self.T, V, n + dn, 1, lng_t=True, lng_v=True, lng_n=True)
            # lngm, lngm_t, lngm_v, lngm_n = self.thermo.lng_ii(self.T, V, n - dn, 1, lng_t=True, lng_v=True, lng_n=True)
            # print(f"lng_n: {lng_n}, {(lngp-lngm)/2/dn[0]}")
            # print(f"lng_Vn: {lng_vn}, {(lngp_v-lngm_v)/2/dn[0]}")
            # print(f"lng_Tn: {lng_tn}, {(lngp_t-lngm_t)/2/dn[0]}")
            # print(f"lng_nn: {lng_nn}, {(lngp_n-lngm_n)/2/dn[0]}")

class saftvrmie_functional(saft_dispersion):
    """

    """

    def __init__(self, N, svrm: saftvrmie, T_red, psi_disp=1.3862, grid_unit=LenghtUnit.ANGSTROM):
        """

        Args:
            N (int): Size of grid
            svrm (saftvrmie): Thermopack object
            T_red (float): Reduced temperature
            psi_disp (float): Width for weighted dispersion density
            grid_unit (LenghtUnit): Unit used for grid
        """
        saft_dispersion.__init__(self,
                                 N,
                                 svrm,
                                 T_red,
                                 psi_disp=psi_disp,
                                 grid_unit=grid_unit)
        self.name = "SAFTVR-MIE"
        self.short_name = "SVRM"


if __name__ == "__main__":
    # Model testing
    pass
