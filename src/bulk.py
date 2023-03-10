#!/usr/bin/env python3
from thermopack.thermopack_state import State
from thermopack.saftvrqmie import saftvrqmie
import numpy as np
from constants import NA, KB, Properties
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))


class Bulk(object):
    """

    """

    def __init__(self,
                 functional,
                 left_state,
                 right_state):
        """Class holding specifications for a gird

        Args:
            geometry (int): PLANAR/POLAR/SPHERICAL
            n_comp (int, optional): Number of components.
            domain_size (float, optional): Sisze of domain. Defaults to 100.0.
            n_grid (int, optional): Number of grid points. Defaults to 1024.
            n_bc (int, optional): Number of boundary points. Defaults to 0.

        Returns:
            None
        """
        self.functional = functional
        self.left_state = left_state
        self.right_state = right_state
        self.particle_diameters, self.particle_diameters_dt = left_state.eos.hard_sphere_diameters(
            left_state.T)
        # Particle radius (Reduced)
        self.R = np.zeros_like(self.particle_diameters)
        self.R[:] = 0.5*self.particle_diameters/functional.grid_reducing_lenght
        self.R_T = np.zeros_like(self.particle_diameters)
        self.R_T[:] = 0.5*self.particle_diameters_dt / \
            functional.grid_reducing_lenght

        # Temperature
        self.temperature = left_state.T
        self.reduced_temperature = self.temperature / \
            functional.thermo.eps_div_kb[0]
        self.beta = 1.0 / self.reduced_temperature
        # Bulk density
        self.reduced_density_left = self.get_reduced_density(
            left_state.partial_density())
        self.reduced_density_right = self.get_reduced_density(
            right_state.partial_density())
        # Bulk fractions
        self.bulk_fractions = self.reduced_density_left / \
            np.sum(self.reduced_density_left)

        # Extract normalized chemical potential (multiplied by beta) (mu/kbT)
        self.mu_res_scaled_beta = self.functional.bulk_excess_chemical_potential(
            self.reduced_density_right)
        self.mu_ig_scaled_beta = np.log(self.reduced_density_right)
        self.mu_scaled_beta = self.mu_ig_scaled_beta + self.mu_res_scaled_beta
        self.real_mu, = functional.thermo.chemical_potential_tv(
            self.temperature, volume=1.0, n=left_state.partial_density())

        # Test
        # mu_res_left, = functional.thermo.chemical_potential_tv(self.temperature, volume=left_state.v, n=left_state.x, property_flag="R")
        # mu_res_left /= (self.temperature*functional.thermo.Rgas)
        # mu_res_right, = functional.thermo.chemical_potential_tv(self.temperature, volume=right_state.v, n=right_state.x, property_flag="R")
        # mu_res_right /= (self.temperature*functional.thermo.Rgas)
        # print("Thermopack mu_res", mu_res_right)
        # print("Functional mu_res", self.mu_res_scaled_beta)
        # print("Diff mu_res", mu_res_right - self.mu_res_scaled_beta)

        # volfac = NA*functional.thermo.sigma[0]**3
        # red_vol = (functional.thermo.sigma[0]/self.functional.grid_reducing_lenght)**3
        # f = left_state.excess_free_energy_density()/(self.temperature*self.functional.thermo.Rgas)
        # a_hs, a_hs_n, = functional.thermo.a_hard_sphere(self.temperature, volume=left_state.v, n=left_state.x, a_n=True)
        # a_disp, a_disp_n, = functional.thermo.a_dispersion(self.temperature, volume=left_state.v, n=left_state.x, a_n=True)
        # print("left a_disp, a_hs, a, a_func", a_disp, a_hs, a_disp + a_hs, f*left_state.v)

        # v_fac = volfac/left_state.v
        # print("left density a_disp, a_hs, a", a_disp*v_fac, a_hs*v_fac, (a_disp + a_hs)*v_fac, f*volfac)
        # print("left density a_disp, a_hs, a", a_disp/left_state.v, a_hs/left_state.v, (a_disp + a_hs)/left_state.v)
        # print("functional left density", functional.bulk_excess_free_energy_density(self.reduced_density_left)*red_vol)
        # mu_disp = a_disp + a_disp_n
        # mu_hs = a_hs + a_hs_n
        # print("left mu_disp, mu_hs", mu_disp, mu_hs)

        # rho = 1.0/right_state.v
        # a_hs, a_hs_v, a_hs_n, = functional.thermo.a_hard_sphere(self.temperature, volume=right_state.v, n=right_state.x, a_n=True, a_v=True)

        # a_disp, a_disp_v, a_disp_n, = functional.thermo.a_dispersion(self.temperature, volume=right_state.v, n=right_state.x, a_n=True, a_v=True)
        # v_fac = volfac/right_state.v
        # print("right density a_disp, a_hs, a", a_disp*v_fac, a_hs*v_fac, (a_disp + a_hs)*v_fac)
        # print("right density a_disp, a_hs, a", a_disp/right_state.v, a_hs/right_state.v, (a_disp + a_hs)/right_state.v)
        # print("functional right density", functional.bulk_excess_free_energy_density(self.reduced_density_right),
        #      functional.bulk_excess_free_energy_density(self.reduced_density_right)*red_vol)
        # mu_disp = a_disp + a_disp_n
        # mu_hs = a_hs + a_hs_n
        # mu_hs *= functional.thermo.m[:]
        # print("right mu_chain", mu_res_right - mu_disp - mu_hs)
        # sys.exit()

    @property
    def red_pressure_right(self):
        # Calculate reduced pressure
        return np.sum(self.reduced_density_right) * self.reduced_temperature * \
            self.functional.bulk_compressibility(self.reduced_density_right)

    @property
    def red_pressure_left(self):
        # Calculate reduced pressure
        return np.sum(self.reduced_density_left) * self.reduced_temperature * \
            self.functional.bulk_compressibility(self.reduced_density_left)

    def get_reduced_density(self, partial_density):
        """
        Calculates the overall number of molecules (reduced) of the system.

        Returns:
            (float): Reduced number of molecules (-)
        """

        reduced_density = np.zeros_like(partial_density)
        reduced_density[:] = partial_density*NA * \
            self.functional.grid_reducing_lenght**3
        return reduced_density

    def get_real_density(self, reduced_density):
        """
        Calculates the overall number of molecules (reduced) of the system.

        Returns:
            (float): Reduced number of molecules (-)
        """

        partial_density = np.zeros_like(reduced_density)
        partial_density[:] = reduced_density / \
            (NA*self.functional.grid_reducing_lenght**3)
        return partial_density

    def update_bulk_densities(self, rho_left, rho_right):
        """
        Calculate bulk states from chemical potential and initial guess for densities
        """
        rho_left_real = self.get_real_density(rho_left)
        rho_left_real = self.functional.thermo.solve_mu_t(
            self.temperature, self.real_mu, rho_initial=rho_left_real)
        self.reduced_density_left[:] = self.get_reduced_density(rho_left_real)

        rho_right_real = self.get_real_density(rho_right)
        rho_right_real = self.functional.thermo.solve_mu_t(
            self.temperature, self.real_mu, rho_initial=rho_right_real)
        self.reduced_density_right[:] = self.get_reduced_density(
            rho_right_real)

        self.bulk_fractions = self.reduced_density_left / \
            np.sum(self.reduced_density_left)

    def get_property(self, prop, reduced_property=True):
        if not reduced_property:
            prop_scaling = 1.0
        else:
            eps = self.functional.thermo.eps_div_kb[0]*KB
            sigma = self.functional.thermo.sigma[0]
            prop_scaling = sigma**3/eps

        if prop == Properties.RHO:
            if reduced_property:
                prop_scaling = sigma**3*NA
            prop_b = prop_scaling*np.column_stack((self.left_state.x/self.left_state.specific_volume(),
                                                   self.right_state.x/self.right_state.specific_volume()))
        elif prop == Properties.FREE_ENERGY:
            prop_b = prop_scaling*np.array([self.left_state.excess_free_energy_density(),
                                            self.right_state.excess_free_energy_density()])
        elif prop == Properties.ENERGY:
            prop_b = prop_scaling*np.array([self.left_state.excess_energy_density(),
                                            self.right_state.excess_energy_density()])
        elif prop == Properties.ENTROPY:
            if reduced_property:
                prop_scaling = sigma**3/KB
            prop_b = prop_scaling*np.array([self.left_state.excess_entropy_density(),
                                            self.right_state.excess_entropy_density()])
        elif prop == Properties.ENTHALPY:
            prop_b = prop_scaling*np.array([self.left_state.excess_enthalpy_density(),
                                            self.right_state.excess_enthalpy_density()])
        elif prop == Properties.CHEMPOT_SUM:
            prop_b = prop_scaling*np.array([np.sum(self.left_state.excess_chemical_potential()*self.left_state.x)/self.left_state.specific_volume(),
                                            np.sum(self.right_state.excess_chemical_potential()*self.right_state.x)/self.right_state.specific_volume()])
        elif prop == Properties.CHEMPOT:
            prop_b = prop_scaling*np.array([self.left_state.excess_chemical_potential()/(self.temperature*self.functional.thermo.Rgas)
                                            + np.log(self.get_reduced_density(self.left_state.x/self.left_state.specific_volume())),
                                            self.right_state.excess_chemical_potential(
            )/(self.temperature*self.functional.thermo.Rgas)
                + np.log(self.get_reduced_density(self.right_state.x/self.right_state.specific_volume()))])
        elif prop == Properties.CHEMPOT_ID:
            prop_b = prop_scaling*np.array([np.log(self.get_reduced_density(self.left_state.x/self.left_state.specific_volume())),
                                            np.log(self.get_reduced_density(self.right_state.x/self.right_state.specific_volume()))])
        elif prop == Properties.CHEMPOT_EX:
            prop_b = prop_scaling*np.array([self.left_state.excess_chemical_potential(),
                                            self.right_state.excess_chemical_potential()])
        elif prop == Properties.PARALLEL_PRESSURE:
            prop_b = prop_scaling * \
                np.array([self.left_state.pressure(),
                         self.right_state.pressure()])

        return prop_b


if __name__ == "__main__":
    pass
