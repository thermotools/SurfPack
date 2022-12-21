#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from dft_numerics import dft_solver
from constants import NA, KB, Geometry, Specification, LenghtUnit, \
    LCOLORS, Properties, get_property_label
from bulk import Bulk
from density_profile import Profile
from grid import Grid
from convolver import Convolver, CurvatureExpansionConvolver
from pyctp.thermopack_state import State, Equilibrium
from weight_functions import ConvType
import numpy as np
import matplotlib.pyplot as plt
from interface import PlanarInterface

# TODO:
# Sign of correlation convolution
# Sign of f conv

class CurvatureExpansionInterface(PlanarInterface):
    """
    Utility class for simplifying specification of PLANAR interface for calculating curvature expansions
    """

    def __init__(self,
                 vle,
                 domain_size=200.0,
                 n_grid=4096,
                 functional_kwargs={}):
        """Class holding specifications for an interface calculation of a planar geometry

        Args:
            vle (Equilibrium): Equilibrium instance
            domain_size (float, optional): Sisze of domain. Defaults to 100.0.
            n_grid (int, optional): Number of grid points. Defaults to 1024.
            functional_kwargs (dict): Optional argiments for functionals. Pass feks.: functional_kwargs={"psi_disp": 1.5}
        Returns:
            None
        """
        PlanarInterface.__init__(self,
                                 Geometry.PLANAR,
                                 thermopack=vle.eos,
                                 temperature=vle.temperature,
                                 domain_size=domain_size,
                                 n_grid=n_grid,
                                 specification=Specification.NUMBER_OF_MOLES,
                                 functional_kwargs=functional_kwargs)
        sc = State.critical(vle.liquid.x)
        self.tanh_profile(vle,sc.T)
        self.profile1 = None
        self.bulk1 = None
        self.profile_diff = None
        self.convolver1 = None
        self.sigma1 = None
        self.tolman = None
        self.helfrich_k = None
        self.helfrich_k_bar = None

    def curvature_residual(self, xvec):
        # Set profile
        n_grid = self.grid.n_grid
        n_c = self.functional.nc
        prof, z = self.unpack_variables(xvec)

        # Specify total equimolar surface
        gamma1 = self.total_adsorption_first_order_expansion()
        prof += np.sum(gamma1)*self.profile_diff

        # Chemical potential
        beta_mu = self.bulk1.mu_scaled_beta[:]

        # Perform convolution integrals
        self.convolver1.convolve_density_profile(prof)

        # Calculate new density profile using the variations of the functional
        res = np.zeros(n_c * n_grid)

        for ic in range(n_c):
            res[ic * n_grid:(ic+1)*n_grid] = xvec[ic * n_grid:(ic+1)*n_grid] \
                - self.profile.densities[ic] * ( beta_mu[ic] - self.convolver.correlation(ic)[:])

        return res

    def solve(self, solver=dft_solver(), log_iter=False):
        # Calculate planar profile
        PlanarInterface.solve(self, solver=solver, log_iter=log_iter)
        if self.converged:
            sigma0 = self.surface_tension(reduced_unit=False)
            self.bulk1 = Bulk.curvature_expansion(self.bulk, sigma0)
            self.profile1 = Profile().copy_profile(self.profile)
            self.profile1.shift_and_scale(shift=0.0,
                                          grid=self.grid,
                                          rho_left=self.bulk1.get_reduced_density(left_state.partial_density()),
                                          rho_right=self.bulk1.get_reduced_density(right_state.partial_density()))

            delta_rho=self.bulk1.get_reduced_density(left_state.partial_density()) - \
                self.bulk1.get_reduced_density(right_state.partial_density())
            self.profile_diff = self.profile.drhodz(grid.z, scaling = 1.0/delta_rho)
            self.convolver1 = CurvatureExpansionConvolver(self.grid,
                                                          self.functional,
                                                          self.bulk.R,
                                                          self.bulk.R_T,
                                                          self.profile)

            # Solve specifying chemical potential
            specification = self.specification
            self.specification = Specification.CHEMICHAL_POTENTIAL
            x0 = self.pack_x_vec(self.profile1)
            x_sol, self.converged, self.n_iter = solver.solve(
                x0, curvature_residual, log_iter)
            if self.converged:
                self.profile1, z = self.unpack_variables(x_sol)
                self.calculate_curvature_expansion_coefficients()
            else:
                print("Interface solver did not converge for rho1")
            self.specification = specification

        return self

    def adsorption_first_order_expansion(self, radius=None):
        # Calculate total adsorption for planar profile
        rho0_left = self.bulk.reduced_density_left
        rho0_right = self.bulk.reduced_density_right
        rho1_left = self.bulk1.reduced_density_left
        rho1_right = self.bulk1.reduced_density_right

        if radius is None:
            radius = self.r_equimolar
        l_left = radius - self.grid.z_edge[0]
        l_right = self.grid.z_edge[-1] - radius

        z = np.zeros_like(self.grid.z)
        z[:] = self.grid.z[:] - radius
        gamma1 = np.zeros_like(rho0_left)
        for ic in range(self.functional.nc):
            gamma1[i] = np.sum(self.profile1.densities[ic][:]*self.grid.integration_weights[:]) \
                - rho1_left[i]*l_left - rho1_right[i]*l_right \
                + 2.0*np.sum(z*self.profile.densities[ic][:]*self.grid.integration_weights[:]) \
                - rho0_left[i]*l_left**2 - rho0_right[i]*l_right**2
        return gamma1

    def calculate_curvature_expansion_coefficients(self):
        g = 2.0
        # Get sigma_0
        _, omega_a = self.grand_potential()
        omega_a += self.bulk.red_pressure_right * self.grid.integration_weights
        sigma_0 = np.sum(omega_a)

        gamma_0 = self.get_adsorption_vector()
        gamma_1 = self.adsorption_first_order_expansion()

        mu_1 = self.bulk1.reduced_temperature * self.bulk1.mu_scaled_beta
        mu_1_c = np.zeros_like(mu_1)
        mu_1_c[:] = 0.5*mu_1
        mu_2 = np.zeros_like(mu_1)
        mu_2_c = np.zeros_like(mu_1)
        # Convolve for (rho_0 * (zw))
        rho_0_zw = self.convolver1.weighted_densities0
        #conv_rho_0_zw = Convolver(self.grid, self.functional, self.bulk.R, self.bulk.R_T)
        # Perform convolution integrals
        #conv_rho_0_zw.convolve_densities_by_type(self.profile, conv_type=ConvType.ZW)
        # Calculate differentials
        f0_zw = self.convolver.get_differential_sum(rho_0_zw)
        sigma_1_0 = 0.5*np.sum(f0_zw*self.grid.integration_weights)
        print("sigma_1_0",sigma_1_0)

        z = np.zeros_like(self.grid.z)
        z[:] = self.grid.z[:] - self.r_equimolar

        print("gamma",self.surface_tension(reduced_unit=False))
        sgn = 1.0 if self.bulk.liquid_is_right() else -1.0
        sigma_1_1 = sgn*np.sum(omega_a*z)*g
        print("sigma_1_1",sigma_1_1)
        self.sigma1 = sigma_1_0 + sigma_1_1
        print("sigma_1",sigma_1)
        print("sigma_0",sigma_0)
        self.tolman = -self.sigma1/sigma_0/g
        print("d_tolman_sphere",self.tolman)

        print("ads",self.get_adsorption_vector(self.r_equimolar))

        delta_rho1 = np.zeros_like(mu_1)
        delta_rho1[:] = self.bulk1.reduced_density_left - self.bulk1.reduced_density_right

        mu_2[:] = ((g - 1)*self.sigma1 - np.sum((gamma0 + 0.5*delta_rho1)*mu_1))/(g*sigma0)*mu_1
        mu_2_c[:] = -(np.sum((gamma0 + 0.5*delta_rho1)*mu_1_c))/sigma0*mu_1_c

        # Convolve for (rho_0 * w_tilde_p)
        conv_rho_0_w_tilde_p = Convolver(self.grid, self.functional, self.bulk.R, self.bulk.R_T)
        # Perform convolution integrals
        conv_rho_0_w_tilde_p.convolve_densities_by_type(self.profile, conv_type=ConvType.TILDEPLUSS)
        f0_w_tilde_p = self.convolver.get_differential_sum(conv_rho_0_w_tilde_p.weighted_densities)

        # Convolve for (rho_0 * w_tilde_m)
        conv_rho_0_w_tilde_m = Convolver(self.grid, self.functional, self.bulk.R, self.bulk.R_T)
        # Perform convolution integrals
        conv_rho_0_w_tilde_m.convolve_densities_by_type(self.profile, conv_type=ConvType.TILDEMINUS)
        f0_w_tilde_m = self.convolver.get_differential_sum(conv_rho_0_w_tilde_m.weighted_densities)

        # (f0 * (zw))
        f0_conv_zw = self.convolver1.comp_differentials0
        # f1
        f1 = self.convolver1.comp_differentials

        # Calculate total adsorption for planar profile
        rho0_left = self.bulk.reduced_density_left
        rho0_right = self.bulk.reduced_density_right

        l_left = self.r_equimolar - self.grid.z_edge[0]
        l_right = self.grid.z_edge[-1] - self.r_equimolar
        rho0_E_z_int = np.zeros_like(rho0_left)
        for ic in range(self.functional.nc):
            rho0_E_z_int[i] = np.sum(z*self.profile.densities[ic][:]*self.grid.integration_weights[:]) \
                - 0.5*rho0_left[i]*l_left**2 - 0.5*rho0_right[i]*l_right**2

        # TODO !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        # SCALING???? beta
        sum_rho1_f0_zw = np.zeros_like(z)
        for ic in range(self.functional.nc):
            sum_rho1_f0_zw[:] -= self.profile1.densities[ic]*f0_conv_zw[ic].corr

        self.helfrich_k = -0.25*np.sum(f0_w_tilde_p*self.grid.integration_weights) \
            - 0.25*np.sum(sum_rho1_f0_zw*self.grid.integration_weights) \
            - 0.25*np.sum(f1*rho_0_zw*self.grid.integration_weights) \
            - 0.5*np.sum(mu_1*rho0_E_z_int) - 2.0*np.sum(mu_2_c*gamma_0) \
            - 0.25*np.sum(mu_1*gamma_1)
        self.helfrich_k_bar = np.sum(omega_a*z*z) \
            + 0.5*np.sum(f0_w_tilde_m*self.grid.integration_weights) \
            - np.sum(f0_zw*z*self.grid.integration_weights) \
            + np.sum((4*mu_2_c - mu_2)*gamma_0)

        # Convert to real units
        eps = self.functional.thermo.eps_div_kb[0] * KB
        self.sigma_1 *= eps / self.functional.grid_reducing_lenght
        self.tolman *= self.functional.grid_reducing_lenght
        self.helfrich_k *= eps
        self.helfrich_k_bar *= eps


    def get_curvature_corrections(self, reduced_unit=False):
        """
        Calculate postion of dividing surface "surface of tension"
        Returns:
        sigma1 (float): First order correction to surface tension
        tolman (float): Tolman lenght
        helfrich_k (float): Helfrich bending rigidity
        helfrich_k_bar (float): Helfrich Gaussian rigidity
        """
        if not self.converged:
            self.print_perform_minimization_message()
            return

        #sigma1 = self.sigma1
        tolman = self.tolman
        helfrich_k = self.helfrich_k
        helfrich_k_bar = self.helfrich_k_bar

        if reduced_unit:
            eps = self.functional.thermo.eps_div_kb[0] * KB
            sigma = self.functional.thermo.sigma[0]
            #sigma1 *= sigma/eps
            tolman /= sigma
            helfrich_k /= eps
            helfrich_k_bar /= eps

        return tolman, helfrich_k, helfrich_k_bar

if __name__ == "__main__":
    pass
