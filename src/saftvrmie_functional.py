#!/usr/bin/env python3
import numpy as np
import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from constants import NA, RGAS, LenghtUnit
from fmt_functionals import bulk_weighted_densities
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
            self.calc_additive_diameters()
            self.calc_non_additive_diameters()
            self.calc_bmcsl_mu()

    def calc_additive_diameters(self):
        """
        """
        self.d_ij = np.zeros((self.nc, self.nc))
        self.d_T_ij = np.zeros((self.nc, self.nc))
        for i in range(self.nc):
            self.d_ij[i,i] = self.d_hs[i]
            self.d_T_ij[i,i] = self.d_T_hs[i]
            for j in range(i+1,self.nc):
                self.d_ij[i,j] = 0.5*(self.d_hs[i]+self.d_hs[j])
                self.d_ij[j,i] = self.d_ij[i,j]
                self.d_T_ij[i,j] = 0.5*(self.d_T_hs[i]+self.d_T_hs[j])
                self.d_T_ij[j,i] = self.d_T_ij[i,j]

        self.d_ij /= self.grid_reducing_lenght
        self.d_T_ij /= self.grid_reducing_lenght

    def calc_non_additive_diameters(self):
        """
        """
        self.delta_ij = np.zeros((self.nc, self.nc))
        self.delta_T_ij = np.zeros((self.nc, self.nc))
        for i in range(self.nc):
            self.delta_ij[i,i] = self.d_ij[i,i]
            self.delta_T_ij[i,i] = self.d_T_ij[i,i]
            for j in range(i+1,self.nc):
                self.delta_ij[i,j], self.delta_T_ij[i,j] = self.thermo.hard_sphere_diameter_ij(i+1, j+1, self.T)
                self.delta_ij[j,i] = self.delta_ij[i,j]
                self.delta_T_ij[j,i] = self.delta_T_ij[i,j]

        self.delta_ij /= self.grid_reducing_lenght
        self.delta_T_ij /= self.grid_reducing_lenght

    def calc_bmcsl_mu(self):
        """
        """
        self.mu_ij = np.zeros((self.nc, self.nc))
        self.mu_ij_T = np.zeros((self.nc, self.nc))
        for i in range(self.nc):
            self.mu_ij[i,i] = 0.5*self.d_hs[i]
            self.mu_ij_T[i,i] = 0.5*self.d_T_hs[i]
            for j in range(i+1,self.nc):
                self.mu_ij[i,j] = self.d_hs[i]*self.d_hs[j]/(self.d_hs[i]+self.d_hs[j])
                self.mu_ij[j,i] = self.mu_ij[i,j]
                self.mu_ij_T[i,j] = (self.d_T_hs[i]*self.d_hs[j] + self.d_hs[i]*self.d_T_hs[j])/(self.d_hs[i]+self.d_hs[j]) \
                    - (self.d_T_hs[i] + self.d_T_hs[j])*self.mu_ij[i,j]/(self.d_hs[i]+self.d_hs[j])
                self.mu_ij_T[j,i] = self.mu_ij_T[i,j]

        self.mu_ij /= self.grid_reducing_lenght
        self.mu_ij_T /= self.grid_reducing_lenght

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
                for j in range(self.nc):
                    n_alpha_j = dens.comp_weighted_densities[j].get_fmt_densities(i)
                    for k in range(j+1,self.nc):
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
        # self.d0[:, :] = 0.0
        # self.d1[:, :] = 0.0
        # self.d2[:, :] = 0.0
        # self.d3[:, :] = 0.0
        # self.d1v[:, :] = 0.0
        # self.d2v[:, :] = 0.0
        # self.mu_disp[:, :] = 0.0
        # return
        if self.na_enabled:
            for i in range(self.n_grid):
                n_alpha = dens.get_fmt_densities(i)
                for j in range(self.nc):
                    n_alpha_j = dens.comp_weighted_densities[j].get_fmt_densities(i)
                    for k in range(j+1,self.nc):
                        n_alpha_k = dens.comp_weighted_densities[k].get_fmt_densities(i)
                        g_jk, g_jk_n, = self.thermo.calc_bmcsl_gij_fmt(n_alpha, self.mu_ij[j,k], calc_g_ij_n=True)
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
                mu_ex[i] += np.sum(dphidn[:4] * bd.dndrho[:, i]) + d0_i[i] * bd.dndrho[0, i]
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
        d0_i = np.zeros(self.nc)
        if self.na_enabled:
            n_alpha = np.zeros(6)
            n_alpha[:4] = bd.n[:]
            for j in range(self.nc):
                for k in range(j+1,self.nc):
                    g_jk, g_jk_n, = self.thermo.calc_bmcsl_gij_fmt(n_alpha, self.mu_ij[j,k], calc_g_ij_n=True)
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
                for j in range(self.nc):
                    n_alpha_j = dens.comp_weighted_densities[j].get_fmt_densities(i)
                    for k in range(j+1,self.nc):
                        n_alpha_k = dens.comp_weighted_densities[k].get_fmt_densities(i)
                        g_jk, g_jk_T, = self.thermo.calc_bmcsl_gij_fmt(n_alpha, self.mu_ij[j,k], mu_ij_T=self.mu_ij_T[j,k])
                        ck = -4*np.pi*n_alpha_j[0]*n_alpha_k[0]
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
            bd = bulk_weighted_densities(rho_b=(n/V)*NA*self.grid_reducing_lenght**3, R=self.R, ms=self.ms)
            n_alpha = np.zeros(6)
            n_alpha[:4] = bd.n[:]
            n_alpha[-1] = -0.5*bd.n[2]
            eps = 1.0e-5
            g_T = np.zeros((self.nc,self.nc))
            for j in range(self.nc):
                for k in range(j, self.nc):
                    g_jk, g_jk_n, g_T[j,k], = self.thermo.calc_bmcsl_gij_fmt(n_alpha,
                                                                             self.mu_ij[j,k],
                                                                             calc_g_ij_n=True,
                                                                             mu_ij_T=self.mu_ij_T[j,k])
                    n_alpha_p = np.zeros(6)
                    n_alpha_m = np.zeros(6)
                    for i in range(6):
                        if abs(n_alpha[i]) > 0.0:
                            n_alpha_p[:] = n_alpha
                            n_alpha_m[:] = n_alpha
                            n_alpha_p[i] += n_alpha[i]*eps
                            n_alpha_m[i] -= n_alpha[i]*eps
                            g_jk_p, = self.thermo.calc_bmcsl_gij_fmt(n_alpha_p,self.mu_ij[j,k])
                            g_jk_m, = self.thermo.calc_bmcsl_gij_fmt(n_alpha_m,self.mu_ij[j,k])
                            print("j,k,alpha",j,k,i,(g_jk_p-g_jk_m)/(2*n_alpha[i]*eps), g_jk_n[i])
            # Temperature differentials
            T = self.T
            self.T = T + T*eps
            self.calc_hs_diameters()
            self.calc_additive_diameters()
            self.calc_non_additive_diameters()
            self.calc_bmcsl_mu()
            g_Tp = np.zeros((self.nc,self.nc))
            for j in range(self.nc):
                for k in range(j, self.nc):
                    g_Tp[j,k], = self.thermo.calc_bmcsl_gij_fmt(n_alpha,self.mu_ij[j,k])
            self.T = T - T*eps
            self.calc_hs_diameters()
            self.calc_additive_diameters()
            self.calc_non_additive_diameters()
            self.calc_bmcsl_mu()
            g_Tm = np.zeros((self.nc,self.nc))
            for j in range(self.nc):
                for k in range(j, self.nc):
                    g_Tm[j,k], = self.thermo.calc_bmcsl_gij_fmt(n_alpha,self.mu_ij[j,k])
            # Reset
            self.T = T
            self.calc_hs_diameters()
            self.calc_additive_diameters()
            self.calc_non_additive_diameters()
            self.calc_bmcsl_mu()
            for j in range(self.nc):
                for k in range(j, self.nc):
                    print("T: j,k",j,k,(g_Tp[j,k]-g_Tm[j,k])/(2*T*eps), g_T[j,k])

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
    eps = 1.0e-5
    T = 25.0
    svrqm = saftvrqmie()
    svrqm.init("H2,Ne",additive_hard_sphere_reference=True)
    svrqm.set_tmin(2.0)
    svrqmf = saftvrqmie_functional(N=1, svrqm=svrqm, T_red=T/svrqm.eps_div_kb[0])
    #
    Tp = T + T*eps
    svrqmfp = saftvrqmie_functional(N=1, svrqm=svrqm, T_red=Tp/svrqm.eps_div_kb[0])
    #
    Tm = T - T*eps
    svrqmfm = saftvrqmie_functional(N=1, svrqm=svrqm, T_red=Tm/svrqm.eps_div_kb[0])
    for i in range(svrqm.nc):
        for j in range(svrqm.nc):
            print("i,j",i,j)
            print("d_ij",(svrqmfp.d_ij[i,j]-svrqmfm.d_ij[i,j])/(2*T*eps), svrqmf.d_T_ij[i,j])
            print("delta_ij",(svrqmfp.delta_ij[i,j]-svrqmfm.delta_ij[i,j])/(2*T*eps), svrqmf.delta_T_ij[i,j])
            print("mu_ij",(svrqmfp.mu_ij[i,j]-svrqmfm.mu_ij[i,j])/(2*T*eps), svrqmf.mu_ij_T[i,j])

    from pyctp.thermopack_state import equilibrium
    vle = equilibrium.bubble_pressure(svrqm, T, z=np.array([0.4,0.6]))
    rho_b = np.zeros_like(vle.liquid.partial_density())
    rho_b[:] = vle.liquid.partial_density()*NA*svrqmf.grid_reducing_lenght**3
    print(rho_b,svrqmf.ms)
    bd = bulk_weighted_densities(rho_b, svrqmf.R, svrqmf.ms)
    phi, dphidn, d0_i = svrqmf.bulk_na_functional_with_differentials(bd)
    for i in range(svrqm.nc):
        n010 = bd.na[0,i]
        bd.na[0,i] = n010 + eps*n010
        phi_p, _, _ = svrqmf.bulk_na_functional_with_differentials(bd)
        bd.na[0,i] = n010 - eps*n010
        phi_m, _, _ = svrqmf.bulk_na_functional_with_differentials(bd)
        bd.na[0,i] = n010
        print(i,(phi_p-phi_m)/(2*eps*n010), d0_i[i])
    z = svrqmf.bulk_compressibility(rho_b)
    print("z thermopack",vle.pressure()*vle.liquid.specific_volume()/(svrqm.Rgas*T))
    print("z functional",z)
    mu = svrqmf.bulk_excess_chemical_potential(rho_b)
    mu_thermopack = vle.liquid.excess_chemical_potential()/(svrqm.Rgas*T)
    print("beta mu excess thermopack",mu_thermopack)
    print("beta mu excess functional",mu)
    print("beta mu excess difference",mu-mu_thermopack)

    x = vle.liquid.x
    v = vle.liquid.v
    a_hs, a_hs_n, = svrqm.a_hard_sphere(T, volume=v, n=x, a_n=True)
    a_disp, a_disp_n, = svrqm.a_dispersion(T, volume=v, n=x, a_n=True)
    mu_disp = a_disp + a_disp_n
    mu_hs = a_hs + a_hs_n
    print("mu_disp, mu_hs", mu_disp, mu_hs)

    # Testing eos
    svrqmf.test_eos_differentials(vle.liquid.specific_volume(), vle.liquid.x)
