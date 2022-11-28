#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from dft_numerics import dft_solver
from constants import NA, KB, Geometry, Specification, LenghtUnit, LCOLORS, Properties, get_property_label
from bulk import Bulk
from density_profile import Profile
from grid import Grid
from convolver import Convolver
from pyctp.pcsaft import pcsaft
from pyctp.saftvrmie import saftvrmie
from pyctp.saftvrqmie import saftvrqmie
from pyctp.ljs_bh import ljs_bh
from pyctp.ljs_wca import ljs_wca, ljs_uv
from pyctp.pets import pets
from pyctp.thermopack_state import state, equilibrium
from pcsaft_functional import pc_saft
from pets_functional import PeTS_functional
from ljs_functional import ljs_bh_functional, ljs_wca_functional, ljs_uv_functional
from saftvrmie_functional import saftvrmie_functional, saftvrqmie_functional
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

class Interface(ABC):
    """

    """

    def __init__(self,
                 geometry,
                 thermopack,
                 temperature,
                 domain_size=100.0,
                 n_grid=1024,
                 specification=Specification.NUMBER_OF_MOLES,
                 functional_kwargs={}):
        """Class holding specifications for an interface calculation

        Args:
            geometry (int): PLANAR/POLAR/SPHERICAL
            thermopack (thermo): Thermopack instance
            temperature (float): Temperature (K)
            domain_size (float, optional): Sisze of domain. Defaults to 100.0.
            n_grid (int, optional): Number of grid points. Defaults to 1024.
            specification (Specification, optional): Override how system of equations are solved
            functional_kwargs (dict): Optional argiments for functionals. Pass feks.: functional_kwargs={"psi_disp": 1.5}
        Returns:
            None
        """
        t_red = temperature/thermopack.eps_div_kb[0]

        # Test functional can be used
        is_fmt_consistent, _ = thermopack.test_fmt_compatibility()
        if not is_fmt_consistent:
            raise AssertionError("thermopack model not compatible with FMT")
        # Create functional
        if isinstance(thermopack, pcsaft):
            self.functional = pc_saft(n_grid, thermopack, t_red, **functional_kwargs)
        elif isinstance(thermopack, ljs_bh):
            self.functional = ljs_bh_functional(n_grid, thermopack, t_red, **functional_kwargs)
        elif isinstance(thermopack, ljs_wca):
            self.functional = ljs_wca_functional(n_grid, thermopack, t_red, **functional_kwargs)
        elif isinstance(thermopack, ljs_uv):
            self.functional = ljs_uv_functional(n_grid, thermopack, t_red, **functional_kwargs)
        elif isinstance(thermopack, saftvrqmie):
            self.functional = saftvrqmie_functional(n_grid, thermopack, t_red, **functional_kwargs)
        elif isinstance(thermopack, saftvrmie):
            self.functional = saftvrmie_functional(n_grid, thermopack, t_red, **functional_kwargs)
        elif isinstance(thermopack, pets):
            self.functional = PeTS_functional(n_grid, thermopack, t_red, **functional_kwargs)
        else:
            raise TypeError("No DFT functional for thermopack model: " + type(thermopack))
        # Set up grid
        self.grid = Grid(geometry, domain_size, n_grid)
        # Set defaults
        self.profile = None
        self.converged = False
        self.v_ext = np.zeros((self.functional.nc, self.grid.n_grid))
        self.bulk = None
        self.specification = specification
        self.n_tot = None
        self.convolver = None
        self.r_equimolar = None
        # Chache for calculated states
        self.s_E = None

    def unpack_variables(self, xvec):
        # Set profile
        n_grid = self.grid.n_grid
        n_c = self.functional.nc
        prof = Profile.empty_profile(n_c, n_grid)
        # Make sure boundary cells are set to bulk densities
        #self.mod_densities.assign_elements(self.densities)
        # Calculate weighted densities
        for ic in range(n_c):
            prof.densities[ic][:] = xvec[ic*n_grid:(ic+1)*n_grid]

        if self.specification == Specification.NUMBER_OF_MOLES:
            z = np.zeros(n_c)
            n_rho = n_c * n_grid
            z[:] = xvec[n_rho:n_rho + n_c]
        else:
            z = None
        return prof, z

    def pack_x_vec(self):
        n_grid = self.grid.n_grid
        n_c = self.functional.nc
        n_rho = n_c * n_grid
        xvec = np.zeros(n_rho + n_c *
                        (1 if self.specification ==
                         Specification.NUMBER_OF_MOLES else 0))
        for ic in range(n_c):
            xvec[ic * n_grid:(ic+1) *
                 n_grid] = self.profile.densities[ic][:]

        if self.specification == Specification.NUMBER_OF_MOLES:
            # Convolution integrals for densities
            self.convolver.convolve_density_profile(self.profile.densities)
            integrals = self.integrate_df_vext()
            exp_beta_mu = np.exp(self.bulk.mu_scaled_beta)
            denum = np.dot(exp_beta_mu, integrals)
            z = self.n_tot / integrals
            xvec[n_rho:n_rho + n_c] = z
        return xvec

    def residual(self, xvec):
        # Set profile
        n_grid = self.grid.n_grid
        n_c = self.functional.nc
        prof, z = self.unpack_variables(xvec)

        n_rho = n_c * n_grid
        beta_mu = np.zeros(n_c)
        beta_mu[:] = self.bulk.mu_scaled_beta[:]

        # Perform convolution integrals
        self.convolver.convolve_density_profile(prof)

        # Calculate new density profile using the variations of the functional
        res = np.zeros(n_rho + n_c *
                       (1 if self.specification ==
                        Specification.NUMBER_OF_MOLES else 0))

        if self.specification == Specification.NUMBER_OF_MOLES:
            exp_beta_mu = np.exp(beta_mu)
            z_grid = np.zeros(n_c)
            integrals = self.integrate_df_vext()
            denum = np.dot(exp_beta_mu, integrals)
            z_grid[:] = self.n_tot/integrals
            res[n_rho:] = z - z_grid
            if not self.grid.geometry == Geometry.PLANAR:
                beta_mu[:] = np.log(z)

        for ic in range(n_c):
            res[ic * n_grid:(ic+1)*n_grid] = - np.exp(self.convolver.correlation(ic)[:]
                       + beta_mu[ic] - self.bulk.beta * self.v_ext[ic][:]) \
                + xvec[ic * n_grid:(ic+1)*n_grid]

        return res

    def solve(self, solver=dft_solver(), log_iter=False):
        if not self.profile:
            self.converged = False
            print("Interface need to be initialized before calling solve")
        else:
            self.reset_cache()
            self.n_tot = self.calculate_total_moles()
            # Set up convolver
            self.convolver = Convolver(self.grid, self.functional, self.bulk.R, self.bulk.R_T)
            x0 = self.pack_x_vec()
            x_sol, self.converged = solver.solve(
                x0, self.residual, log_iter)
            if self.converged:
                self.profile, z = self.unpack_variables(x_sol)
                # Update bulk properties
                if self.specification == Specification.NUMBER_OF_MOLES:
                    if not self.grid.geometry == Geometry.PLANAR:
                        # Chemical potential can have changed
                        beta_mu = np.log(z)
                        self.bulk.update_chemical_potential(beta_mu)
                        rho_left = np.zeros_like(self.bulk.real_mu)
                        rho_right = np.zeros_like(self.bulk.real_mu)
                        for i in range(self.functional.nc):
                            rho_left[i] = self.profile.densities[i][1]
                            rho_right[i] = self.profile.densities[i][-2]
                        self.bulk.update_densities(rho_left, rho_right)
                self.calculate_equimolar_dividing_surface()
            else:
                print("Interface solver did not converge")
        return self

    def reset_cache(self):
        # Reset cached states
        self.s_E = None

    def single_convolution(self):
        # Set up convolver?
        if not self.convolver:
            self.convolver = Convolver(self.grid, self.functional, self.bulk.R, self.bulk.R_T)
        # Reset cache
        self.reset_cache()
        # Perform convolution integrals
        self.convolver.convolve_density_profile(self.profile)

    def tanh_profile(self, vle, t_crit, rel_pos_dividing_surface=0.5, invert_states=False):
        """
        Initialize tangens hyperbolicus profile

            rel_pos_dividing_surface (float, optional): Relative location of initial dividing surface. Default value 0.5.
        Returns:
            (float): Grand potential
            (array): Grand potential contribution for each grid point
        """

        left_state = vle.vapor
        right_state = vle.liquid
        if invert_states:
            left_state, right_state = right_state, left_state

        self.bulk = Bulk(self.functional, left_state, right_state)

        # Calculate chemical potential (excess + ideal)
        reduced_temperature = min(vle.temperature/t_crit, 1.0)

        self.profile = Profile.tanh_profile(self.grid,
                                            self.bulk,
                                            reduced_temperature,
                                            rel_pos_dividing_surface=rel_pos_dividing_surface)
        return self

    def constant_profile(self, state):
        """
        Initialize constant density profiles. Correct for external potential if present.

        Returns:
            state (State): Thermodynamic state
        """

        # Calculate chemical potential (excess + ideal)
        self.bulk = Bulk(self.functional, state, state)
        self.profile = Profile.constant_profile(self.grid, self.bulk, self.v_ext)
        return self

    def set_profile(self, vle, profile, invert_states=False):
        """
        Initialize using excisting profile

        Returns:
            state (State): Thermodynamic state
        """
        left_state = vle.vapor
        right_state = vle.liquid
        if invert_states:
            left_state, right_state = right_state, left_state

        self.bulk = Bulk(self.functional, left_state, right_state)
        self.profile = Profile()
        self.profile.copy_profile(profile)
        self.converged = True
        return self

    def grand_potential(self):
        """
        Calculates the grand potential in the system.

        Returns:
            (float): Grand potential
            (array): Grand potential contribution for each grid point
        """

        # Calculate chemical potential (excess + ideal)
        mu = self.bulk.reduced_temperature * self.bulk.mu_scaled_beta

        # FMT hard-sphere part
        omega_a = self.bulk.reduced_temperature * \
            self.functional.excess_free_energy(
                self.convolver.weighted_densities)

        # Add ideal part and extrinsic part
        for i in range(self.functional.nc):
            # Ideal part
            omega_a[:] += self.bulk.reduced_temperature * self.profile.densities[i][:] * \
                (np.log(self.profile.densities[i][:]) - 1.0)
            # Extrinsic part
            omega_a[:] += self.profile.densities[i][:] \
                * (self.v_ext[i][:] - mu[i])

        omega_a[:] *= self.grid.integration_weights

        # Integrate
        omega = np.sum(omega_a[:])

        return omega, omega_a

    def excess_free_energy(self):
        """
        Calculates the excess free energy in the system.

        Returns:
            (array): Excess free energy (-)
        """
        if not self.profile:
            print("Need profile to calculate excess_free_energy")
            return None

        A_E = self.bulk.reduced_temperature * \
            self.functional.excess_free_energy(
                self.convolver.weighted_densities)

        return A_E

    def sum_rho_excess_chemical_potential(self):
        """
        Calculates the excess chemical potential.

        Returns:
            (array): Excess chemical potential ()
        """
        if not self.profile:
            print("Need profile to calculate sum_rho_excess_chemical_potential")
            return None

        mu_E = np.zeros(self.grid.n_grid)
        # Subtract ideal part from overall mu
        for i in range(self.functional.nc):
            # Ideal part
            mu_E[:] += self.bulk.reduced_temperature * self.profile.densities[i][:] * \
                (self.bulk.mu_scaled_beta[i] - np.log(self.profile.densities[i][:]))
        return mu_E

    @abstractmethod
    def surface_tension(self, reduced_unit=False):
        """
        Calculates the surface tension of the system.

        Args;
            reduced_unit (bool): Calculate using reduced units? Default False.

        Returns:
            (float): Surface tension (reduced units or J/m2)
        """
        pass

    def surface_tension_real_units(self):
        """
        Calculates the surface tension of the system.

        Returns:
            (float): Surface tension (J/m2)
        """
        gamma_star = self.surface_tension(reduced_unit=True)
        eps = self.functional.thermo.eps_div_kb[0] * KB
        sigma = self.functional.thermo.sigma[0]
        gamma = gamma_star * eps / sigma ** 2
        return gamma

    @abstractmethod
    def parallel_pressure(self):
        """
        Calculates the parallel component of the pressure tensor

        Returns:
            (float): Pressure
        """

        return None

    def calculate_total_moles(self):
        """
        Calculates the overall moles of the system.

        Returns:
            (float): Number of moles (mol)
        """

        n_tot = 0.0
        for ic in range(self.functional.nc):
            n_tot += np.sum(self.profile.densities[ic][:]*self.grid.integration_weights[:])
        return n_tot

    def integrate_df_vext(self):
        """
        Calculates the integral of exp(-beta(df+Vext)).

        Returns:
            (float): Integral (-)
        """
        n_c = self.functional.nc
        integral = np.zeros(n_c)
        for ic in range(n_c):
            integral[ic] = np.sum(self.grid.integration_weights*
                                  np.exp(self.convolver.correlation(ic)[:]
                                         - self.bulk.beta * self.v_ext[ic][:]))
        return integral

    def calculate_equimolar_dividing_surface(self):
        """

        """
        if not self.converged:
            self.print_perform_minimization_message()
            return

        rho1 = np.sum(self.bulk.reduced_density_left)
        rho2 = np.sum(self.bulk.reduced_density_right)

        N = self.calculate_total_moles()
        if self.grid.geometry == Geometry.PLANAR:
            V = self.grid.domain_size
            prefac = 1.0
            exponent = 1.0
        elif self.grid.geometry == Geometry.SPHERICAL:
            prefac = 4*np.pi/3
            V = prefac*self.grid.domain_size**3
            exponent = 1.0/3.0
        elif self.grid.geometry == Geometry.POLAR:
            prefac = np.pi
            V = self.grid.domain_size**2
            exponent = 0.5

        self.r_equimolar = ((N - V*rho2)/(rho1 - rho2)/prefac)**exponent
        #print("Re, Re/R", R, R/self.grid.domain_size)

    def get_adsorption_vector(self, radius):
        """
        Get adsoprption vector Gamma (Gamma is zero for equimolar radius)
        """
        if not self.profile:
            print("Need profile to calculate adsorption vector")
            return

        iR = self.grid.get_index_of_rel_pos(radius/self.grid.domain_size)
        w_left = self.grid.get_left_weight(radius)
        w_right = self.grid.integration_weights[iR] - w_left
        gamma = np.zeros_like(self.bulk.reduced_density_left)
        for i in range(self.functional.nc):
            gamma[i] += w_left*(self.profile.densities[i][iR] - self.bulk.reduced_density_left[i])
            gamma[i] += w_right*(self.profile.densities[i][iR] - self.bulk.reduced_density_right[i])
            for j in range(iR):
                gamma[i] += self.grid.integration_weights[j]*(self.profile.densities[i][j] - self.bulk.reduced_density_left[i])
            for j in range(iR+1,self.grid.n_grid):
                gamma[i] += self.grid.integration_weights[j]*(self.profile.densities[i][j] - self.bulk.reduced_density_right[i])
        return gamma

    def get_excess_free_energy_density(self, reduced=True):
        """
        Calculates the Helmholtz energy in the system.

        Returns:
            (array): Helmholtz energy for each grid point (J/m3 or dimensionless)
        """
        if not self.profile:
            print("Need profile to calculate free energy density")
            return None

        F = self.functional.excess_free_energy(self.convolver.weighted_densities)
        if reduced:
            F *= self.bulk.reduced_temperature*(self.functional.thermo.sigma[0]/self.functional.grid_reducing_lenght)**3
        else:
            F *= KB*self.bulk.temperature/self.functional.grid_reducing_lenght**3
        return F

    def get_excess_chemical_potential_density_sum(self):
        """
        Calculates the sum of excess chemical potential per volume.

        Returns:
            (array): Excess chemical potential ()
        """
        if not self.profile:
            print("Need profile to calculate chemical potential density")
            return None

        # Calculate the excess chemical potential
        mu_E = self.sum_rho_excess_chemical_potential()
        mu_E *= (self.functional.thermo.sigma[0]/self.functional.grid_reducing_lenght)**3
        return mu_E

    def get_excess_enthalpy_density(self, reduced=True):
        """
        Calculates the excess enthalpy density in reduced units

        Returns:
            (array): Excess enthalpy
        """
        if not self.profile:
            print("Need profile to calculate enthalpy density")
            return None

        s_E = self.get_excess_entropy_density()
        sum_rho_mu_E = self.get_excess_chemical_potential_density_sum()
        h_E = np.zeros(self.grid.n_grid)
        h_E[:] = self.bulk.reduced_temperature*s_E[:] + sum_rho_mu_E[:]
        if not reduced:
            h_E[:] *= self.functional.thermo.eps_div_kb[0]*KB/self.functional.thermo.sigma[0]**3
        return h_E

    def get_excess_energy_density(self, reduced=True):
        """
        Calculates the excess energy density in reduced units

        Returns:
            (array): Excess energy density
        """
        if not self.profile:
            print("Need profile to calculate energy density")
            return None
        a_E = self.get_excess_free_energy_density()
        s_E = self.get_excess_entropy_density()
        u_E = a_E + s_E*self.bulk.reduced_temperature
        if not reduced:
            u_E[:] *= self.functional.thermo.eps_div_kb[0]*KB/self.functional.thermo.sigma[0]**3
        return u_E

    def get_excess_entropy_density(self, reduced=True):
        """
        Get reduced entropy per reduced volume (dimensionless or J/m3/K)
        """
        if not self.profile:
            print("Need profile to calculate entropy density")
            return None

        if self.s_E is None:
            f = self.functional.excess_free_energy(self.convolver.weighted_densities)
            f_T = self.convolver.functional_temperature_differential_convolution(self.profile.densities)
            vol_fac = (self.functional.thermo.sigma[0]/self.functional.grid_reducing_lenght)**3
            s = (- f - self.bulk.temperature * f_T)*vol_fac
            self.s_E = s
        if reduced:
            scaling = 1.0
        else:
            scaling = self.functional.thermo.Rgas/(NA*self.functional.thermo.sigma[0]**3)
        return self.s_E*scaling

    def chemical_potential(self, ic=0, properties="IE", reduced=True):
        """
        Get chemical potential (J/mol)
        Args:
           ic (int): Component index
        """
        mu = np.zeros(self.grid.n_grid)
        if "E" in properties:
            mu[:] -= self.convolver.correlation(ic)[:]
        if "I" in properties:
            mu[:] += np.log(self.profile.densities[ic][:])
        return mu

    def print_perform_minimization_message(self):
        """

        """
        print('A successful minimisation have not been yet converged, and the equilibrium profile is missing.')
        print('Please perform a minimisation before performing result operations.')

    def generate_case_name(self):
        """
        Generate case name from specifications
        """
        return f'{self.grid.geometry.name}_{"{:.3f}".format(self.bulk.temperature)}_{self.functional.short_name}'

    def save_equilibrium_density_profile(self):
        """
        Save equilibrium density profile to file
        """
        if not self.converged:
            self.print_perform_minimization_message()
            return

        nd_densities = self.profiles.densities.get_nd_copy()
        filename = self.generate_case_name() + '.dat'
        np.savetxt(filename,
                   np.c_[self.grid.r[:],
                         nd_densities[:, :],
                         (nd_densities[:, :].T / self.cDFT.bulk_densities).T],
                   header="# r, rho, rho/rho_bulk")

    def get_property_profiles(self,
                              prop=Properties.RHO,
                              reduced_property=True):
        """
        Get equilibrium profile
        Args:
            data_dict: Additional data to plot
        """
        # if not self.converged:
        #     self.print_perform_minimization_message()
        #     return

        prop_profiles = []
        legend = []
        unit_scaling = 1.0
        if prop == Properties.GRID:
            if self.functional.grid_unit == LenghtUnit.ANGSTROM and reduced_property:
                prop_fac = 1.0/(self.functional.thermo.sigma[0]*1e10)
                label = r"$z^*$"
            else:
                prop_fac = self.functional.grid_reducing_lenght*1e10
                label = r"$z$ (Å)"
            prop_profiles.append(self.grid.z[:]*prop_fac)
        elif prop == Properties.RHO:
            if reduced_property:
                prop_fac = (self.functional.thermo.sigma[0]/self.functional.grid_reducing_lenght)**3
                label = r"$\rho^*$"
            else:
                prop_fac = 1.0e-3/(NA*self.functional.grid_reducing_lenght**3)
                label = r"$\rho$ (kmol/m$^3$)"
            for i in range(self.profile.densities.nc):
                prop_profiles.append(self.profile.densities[i][:] * prop_fac)
                legend.append(f"{self.functional.thermo.get_comp_name(i+1)}")
        else:
            legend.append("Functional")
            if reduced_property:
                prop_fac = 1.0
            else:
                eps = self.functional.thermo.eps_div_kb[0]*KB
                sigma = self.functional.thermo.sigma[0]
                prop_fac = eps/sigma**3
            if prop == Properties.FREE_ENERGY:
                prop_val = self.get_excess_free_energy_density()
                if reduced_property:
                    label = r"$a^*_{\rm{E}}$"
                else:
                    unit_scaling = 1.0e-3
                    prop_fac = 1.0e-3
                    label = r"$a_{\rm{E}}$ (kJ/mol)"
            elif prop == Properties.ENERGY:
                prop_val = self.get_excess_energy_density()
                if reduced_property:
                    label = r"$u^*_{\rm{E}}$"
                else:
                    unit_scaling = 1.0e-3
                    prop_fac = 1.0e-3
                    label = r"$u_{\rm{E}}$ (kJ/mol)"
            elif prop == Properties.ENTROPY:
                prop_val = self.get_excess_entropy_density()
                if reduced_property:
                    label = r"$s^*_{\rm{E}}$"
                else:
                    prop_fac = NA*KB
                    label = r"$s_{\rm{E}}$ (J/mol/K)"
            elif prop == Properties.ENTHALPY:
                prop_val = self.get_excess_enthalpy_density()
                if reduced_property:
                    label = r"$h^*_{\rm{E}}$"
                else:
                    unit_scaling = 1.0e-3
                    prop_fac = 1.0e-3
                    label = r"$h_{\rm{E}}$ (kJ/mol)"
            elif prop == Properties.CHEMPOT_SUM:
                prop_val = self.get_excess_chemical_potential_density_sum()
                if reduced_property:
                    label = r"$\mu^*_{\rm{E}}$"
                else:
                    unit_scaling = 1.0e-3
                    prop_fac = 1.0e-3
                    label = r"$\mu_{\rm{E}}$ (kJ/mol)"
            elif prop == Properties.PARALLEL_PRESSURE:
                prop_val = self.parallel_pressure()
                if reduced_property:
                    label = r"$p^*_{\parallel}$"
                else:
                    unit_scaling = 1.0e-6
                    prop_fac = 1.0e-6
                    label = r"$p_{\parallel}$ (MPa)"
            prop_profiles.append(prop_val * prop_fac)

        return prop_profiles, legend, label, unit_scaling

    def plot_property_profiles(self,
                               prop=Properties.RHO,
                               xlim=None,
                               ylim=None,
                               plot_reduced_property=True,
                               plot_equimolar_surface=False,
                               plot_bulk=False,
                               include_legend=False,
                               continue_after_plotting=False):
        """
        Plot equilibrium profiles
        Args:
            data_dict: Additional data to plot
        """
        # if not self.converged:
        #     self.print_perform_minimization_message()
        #     return
        z, _, xlabel, _ = self.get_property_profiles(prop=Properties.GRID,
                                                     reduced_property=plot_reduced_property)
        z = z[0]
        prop_profiles, legend, ylabel, unit_scaling = self.get_property_profiles(prop=prop,
                                                                                 reduced_property=plot_reduced_property)
        # prop_b_scaling = 1.0
        # if not grid_unit:
        #     grid_unit = self.functional.grid_unit
        # fig, ax = plt.subplots(1, 1)
        # if grid_unit==LenghtUnit.REDUCED:
        #     ax.set_xlabel(r"$z/\sigma_{11}$")
        # elif grid_unit==LenghtUnit.ANGSTROM:
        #     ax.set_xlabel(r"$z$ (Å)")
        # if self.functional.grid_unit == LenghtUnit.ANGSTROM and grid_unit == LenghtUnit.REDUCED:
        #     len_fac = 1.0/(self.functional.thermo.sigma[0]*1e10)
        # else:
        #     len_fac = self.functional.grid_reducing_lenght*1e10

        # if prop == Properties.RHO:
        #     dens_fac = np.ones(self.profile.densities.nc)
        #     if plot_reduced_property:
        #         dens_fac *= (self.functional.thermo.sigma[0]/self.functional.grid_reducing_lenght)**3
        #         ax.set_ylabel(r"$\rho^*$")
        #     else:
        #         prop_b_scaling = 1.0e-3
        #         dens_fac *= 1.0e-3/(NA*self.functional.grid_reducing_lenght**3)
        #         ax.set_ylabel(r"$\rho$ (kmol/m$^3$)")
        #     for i in range(self.profile.densities.nc):
        #         ax.plot(self.grid.z[:]*len_fac,
        #                 self.profile.densities[i][:] * dens_fac[i],
        #                 lw=2, color=LCOLORS[i],
        #                 label=f"{self.functional.thermo.get_comp_name(i+1)}")
        # else:
        #     label = "Functional"
        #     if plot_reduced_property:
        #         prop_scaling = 1.0
        #     else:
        #         eps = self.functional.thermo.eps_div_kb[0]*KB
        #         sigma = self.functional.thermo.sigma[0]
        #         prop_scaling = eps/sigma**3
        #     if prop == Properties.FREE_ENERGY:
        #         prop_val = self.get_excess_free_energy_density()
        #         if plot_reduced_property:
        #             ax.set_ylabel(r"$a^*_{\rm{E}}$")
        #         else:
        #             prop_b_scaling = 1.0e-3
        #             prop_scaling *= prop_b_scaling
        #             ax.set_ylabel(r"$a_{\rm{E}}$ (kJ/mol)")
        #     elif prop == Properties.ENERGY:
        #         prop_val = self.get_excess_energy_density()
        #         if plot_reduced_property:
        #             ax.set_ylabel(r"$u^*_{\rm{E}}$")
        #         else:
        #             prop_b_scaling = 1.0e-3
        #             prop_scaling *= prop_b_scaling
        #             ax.set_ylabel(r"$u_{\rm{E}}$ (kJ/mol)")
        #     elif prop == Properties.ENTROPY:
        #         prop_val = self.get_excess_entropy_density()
        #         if plot_reduced_property:
        #             ax.set_ylabel(r"$s^*_{\rm{E}}$")
        #         else:
        #             prop_scaling = NA*KB
        #             ax.set_ylabel(r"$s_{\rm{E}}$ (J/mol/K)")
        #     elif prop == Properties.ENTHALPY:
        #         prop_val = self.get_excess_enthalpy_density()
        #         if plot_reduced_property:
        #             ax.set_ylabel(r"$h^*_{\rm{E}}$")
        #         else:
        #             prop_b_scaling = 1.0e-3
        #             prop_scaling *= prop_b_scaling
        #             ax.set_ylabel(r"$h_{\rm{E}}$ (kJ/mol)")
        #     elif prop == Properties.CHEMPOT_SUM:
        #         prop_val = self.get_excess_chemical_potential_density_sum()
        #         if plot_reduced_property:
        #             ax.set_ylabel(r"$\mu^*_{\rm{E}}$")
        #         else:
        #             prop_scaling *= prop_b_scaling
        #             ax.set_ylabel(r"$\mu_{\rm{E}}$ (kJ/mol)")
        #     elif prop == Properties.PARALLEL_PRESSURE:
        #         prop_val = self.parallel_pressure()
        #         if plot_reduced_property:
        #             ax.set_ylabel(r"$p^*_{\parallel}$")
        #         else:
        #             prop_b_scaling = 1.0e-6
        #             prop_scaling *= prop_b_scaling
        #             ax.set_ylabel(r"$p_{\parallel}$ (MPa)")

        #     ax.plot(self.grid.z[:]*len_fac,
        #             prop_val*prop_scaling,
        #             label=label,
        #             lw=2, color=LCOLORS[0])

        fig, ax = plt.subplots(1, 1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        for i in range(len(prop_profiles)):
            ax.plot(z, prop_profiles[i],
                    lw=2, color=LCOLORS[i],
                    label=legend[i])

        if plot_bulk:
            z_b = np.array([z[0], z[-1]])
            prop_b = self.bulk.get_property(prop, reduced_property=plot_reduced_property)
            if prop == Properties.RHO:
                for i in range(self.profile.densities.nc):
                    ax.plot(z_b, prop_b[i,:]*unit_scaling, color=LCOLORS[i], marker="o", linestyle="None")
            else:
                ax.plot(z_b, prop_b*unit_scaling, color=LCOLORS[0], marker="o", linestyle="None")

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        if plot_equimolar_surface:
            # Plot equimolar dividing surface
            yl = ax.get_ylim()
            r_equimolar = self.r_equimolar*z[-1]/self.grid.z[-1]
            ax.plot([r_equimolar, r_equimolar],
                    yl,
                    lw=1, color="k",
                    linestyle="--",
                    label="Eq. mol. surf.")

        if include_legend:
            leg = plt.legend(loc="best", numpoints=1, frameon=False)

        filename = self.generate_case_name() + "_" + prop.name + ".pdf"
        plt.savefig(filename)

        if continue_after_plotting:
            plt.draw()
        else:
            plt.show()

    def test_grand_potential_bulk(self):
        """
        """
        # Test grand potential for bulk phase
        _, omega_a = self.grand_potential()
        omega_a *= 1/self.grid.integration_weights
        reducing = self.functional.grid_reducing_lenght**3 / (self.functional.thermo.eps_div_kb[0] * KB)
        print("Right state:")
        print(f"  omega={omega_a[-1]}")
        print(f"  pressure: {self.bulk.red_pressure_right}")
        print(f"  pressure + omega: {self.bulk.red_pressure_right + omega_a[-1]}")
        print(f"  thermopack pressure: {self.bulk.right_state.pressure()*reducing}")
        print("Left state:")
        print(f"  omega={omega_a[0]}")
        print(f"  pressure: {self.bulk.red_pressure_left}")
        print(f"  pressure + omega: {self.bulk.red_pressure_left + omega_a[0]}")
        print(f"  thermopack pressure: {self.bulk.left_state.pressure()*reducing}")

    def test_functional_differential(self, alias, eps=1.0e-5, ic=0):
        """ Method intended for debugging functional differentials
        Args:
        alias (string): Name of weigthed density
        eps (float): Size and direction of relative perturbation
        ic (int): If set, the local density will be set.
        """
        self.single_convolution()
        # Evaluate differential
        self.convolver.functional.differentials(self.convolver.weighted_densities)
        diff = np.zeros(self.grid.n_grid)
        if alias == "rho":
            diff[:] = self.functional.mu_of_rho[:, ic]
        else:
            diff[:] = self.functional.diff[alias][:, ic]
        # Perturbate density
        n,ni = self.convolver.weighted_densities.perturbate(alias=alias,eps=-eps,ic=ic)
        n0 = np.zeros_like(n)
        n0[:] = n
        Fm = self.functional.excess_free_energy(self.convolver.weighted_densities)
        # Reset density
        self.convolver.weighted_densities.set_density(n,ni,alias=alias,ic=ic)
        # Perturbate density
        n,ni = self.convolver.weighted_densities.perturbate(alias=alias,eps=eps,ic=ic)
        Fp = self.functional.excess_free_energy(self.convolver.weighted_densities)
        # Reset density
        self.convolver.weighted_densities.set_density(n,ni,alias=alias,ic=ic)
        # Plot differentials
        n_div = n if ni is None else ni
        plt.plot(self.grid.z,(Fp - Fm)/(2*n_div*eps),label="Numeric")
        plt.plot(self.grid.z,diff,label="Analytic")
        plt.xlabel("$z$")
        plt.title("Numerical test of differential for: " + alias + " of component " + str(ic))
        leg = plt.legend(loc="best", numpoints=1, frameon=False)
        plt.show()

    def test_functional_in_bulk(self, reduced=True):
        """ Plot bulk and functional values together
        """
        # Make sure convlver is set up
        self.single_convolution()

        sigma = self.functional.thermo.sigma[0]
        eps = self.functional.thermo.eps_div_kb[0]*KB
        Rgas = self.functional.thermo.Rgas
        if reduced:
            len_fac = self.functional.grid_reducing_lenght/self.functional.thermo.sigma[0]
            x_label = r"$z/\sigma$"
            energy_scaling = sigma**3/eps
            s_scaling = (NA*sigma**3)/Rgas
            p_scaling = sigma**3/eps
        else:
            if self.functional.grid_reducing_lenght == self.functional.thermo.sigma[0]:
                len_fac = self.functional.thermo.sigma[0]*1e10
            x_label = r"$z$ (Å)"
            energy_scaling = 1.0
            s_scaling = 1.0
            p_scaling = 1.0e-6 # Pa -> MPa

        s_E = self.get_excess_entropy_density(reduced)
        a_E = self.get_excess_free_energy_density(reduced)
        p = self.parallel_pressure(reduced)
        h_E = self.get_excess_enthalpy_density(reduced)
        u_E = self.get_excess_energy_density(reduced)

        plt.figure()
        plt.plot(self.grid.z*len_fac, s_E,label=r"Functional")
        plt.plot([self.grid.z[0]*len_fac], s_scaling*np.array([self.bulk.left_state.excess_entropy_density()]),
                 label=r"Bulk left", linestyle="None", marker="o")
        plt.plot([self.grid.z[-1]*len_fac], s_scaling*np.array([self.bulk.right_state.excess_entropy_density()]),
                 label=r"Bulk right", linestyle="None", marker="o")
        plt.ylabel(get_property_label(Properties.ENTROPY, reduced))
        plt.xlabel(x_label)
        leg = plt.legend(loc="best", numpoints=1, frameon=False)

        plt.figure()
        plt.plot(self.grid.z*len_fac, a_E,label=r"Functional")
        plt.plot([self.grid.z[0]*len_fac], energy_scaling*np.array([self.bulk.left_state.excess_free_energy_density()]),
                 label=r"Bulk left", linestyle="None", marker="o")
        plt.plot([self.grid.z[-1]*len_fac], energy_scaling*np.array([self.bulk.right_state.excess_free_energy_density()]),
                 label=r"Bulk right", linestyle="None", marker="o")
        plt.ylabel(get_property_label(Properties.FREE_ENERGY, reduced))
        plt.xlabel(x_label)
        leg = plt.legend(loc="best", numpoints=1, frameon=False)

        mu_b = self.bulk.get_property(Properties.CHEMPOT, reduced_property=False)
        for ic in range(self.functional.nc):
            mu = self.chemical_potential(ic,"IE")
            plt.figure()
            plt.plot(self.grid.z*len_fac, mu,label=r"Functional")
            plt.plot([self.grid.z[0]*len_fac], mu_b[0][ic], label=r"Bulk left", linestyle="None", marker="o")
            plt.plot([self.grid.z[-1]*len_fac], mu_b[1][ic], label=r"Bulk right", linestyle="None", marker="o")
            plt.ylabel(get_property_label(Properties.CHEMPOT, reduced, ic))
            plt.xlabel(x_label)
            leg = plt.legend(loc="best", numpoints=1, frameon=False)

        plt.figure()
        plt.plot(self.grid.z*len_fac, p,label=r"Functional")
        plt.plot([self.grid.z[0]*len_fac], p_scaling*np.array([self.bulk.left_state.pressure()]),
                 label=r"Bulk left", linestyle="None", marker="o")
        plt.plot([self.grid.z[-1]*len_fac], p_scaling*np.array([self.bulk.right_state.pressure()]),
                 label=r"Bulk right", linestyle="None", marker="o")
        plt.ylabel(get_property_label(Properties.PARALLEL_PRESSURE, reduced))
        plt.xlabel(x_label)
        leg = plt.legend(loc="best", numpoints=1, frameon=False)

        plt.figure()
        plt.plot(self.grid.z*len_fac, u_E,label=r"Functional")
        plt.plot([self.grid.z[0]*len_fac], energy_scaling*np.array([self.bulk.left_state.excess_energy_density()]),
                 label=r"Bulk liquid", linestyle="None", marker="o")
        plt.plot([self.grid.z[-1]*len_fac], energy_scaling*np.array([self.bulk.right_state.excess_energy_density()]),
                 label=r"Bulk vapour", linestyle="None", marker="o")
        plt.ylabel(get_property_label(Properties.ENERGY, reduced))
        plt.xlabel(x_label)
        leg = plt.legend(loc="best", numpoints=1, frameon=False)

        plt.figure()
        plt.plot(self.grid.z*len_fac, h_E,label=r"Functional")
        plt.plot([self.grid.z[0]*len_fac], energy_scaling*np.array([self.bulk.left_state.excess_enthalpy_density()]),
                 label=r"Bulk liquid", linestyle="None", marker="o")
        plt.plot([self.grid.z[-1]*len_fac], energy_scaling*np.array([self.bulk.right_state.excess_enthalpy_density()]),
                 label=r"Bulk vapour", linestyle="None", marker="o")
        plt.ylabel(get_property_label(Properties.ENTHALPY, reduced))
        plt.xlabel(x_label)
        leg = plt.legend(loc="best", numpoints=1, frameon=False)
        plt.show()


class PlanarInterface(Interface):
    """
    Utility class for simplifying specification of PLANAR interface
    """

    def __init__(self,
                 thermopack,
                 temperature,
                 domain_size=100.0,
                 n_grid=1024,
                 specification = Specification.NUMBER_OF_MOLES,
                 functional_kwargs={}):
        """Class holding specifications for an interface calculation of a planar geometry

        Args:
            thermopack (thermo): Thermopack instance
            temperature (float): Temperature (K)
            domain_size (float, optional): Sisze of domain. Defaults to 100.0.
            n_grid (int, optional): Number of grid points. Defaults to 1024.
            specification (Specification, optional): Override how system of equations are solved
            functional_kwargs (dict): Optional argiments for functionals. Pass feks.: functional_kwargs={"psi_disp": 1.5}
        Returns:
            None
        """
        Interface.__init__(self,
                           Geometry.PLANAR,
                           thermopack=thermopack,
                           temperature=temperature,
                           domain_size=domain_size,
                           n_grid=n_grid,
                           specification=specification,
                           functional_kwargs=functional_kwargs)

    @staticmethod
    def from_tanh_profile(vle,
                          t_crit,
                          domain_size=100.0,
                          n_grid=1024,
                          rel_pos_dividing_surface=0.5,
                          invert_states=False,
                          functional_kwargs={}):
        """
        Initialize tangens hyperbolicus profile

            rel_pos_dividing_surface (float, optional): Relative location of initial dividing surface. Default value 0.5.
        Returns:
            (float): Grand potential
            (array): Grand potential contribution for each grid point
            functional_kwargs (dict): Optional argiments for functionals. Pass feks.: functional_kwargs={"psi_disp": 1.5}

        """
        pif = PlanarInterface(thermopack=vle.eos,
                              temperature=vle.temperature,
                              domain_size=domain_size,
                              n_grid=n_grid,
                              functional_kwargs=functional_kwargs)
        pif.tanh_profile(vle,
                         t_crit,
                         rel_pos_dividing_surface=rel_pos_dividing_surface,
                         invert_states=invert_states)
        return pif

    @staticmethod
    def from_profile(vle,
                     profile,
                     domain_size=100.0,
                     n_grid=1024,
                     invert_states=False,
                     functional_kwargs={}):
        """
        Initialize tangens hyperbolicus profile

            rel_pos_dividing_surface (float, optional): Relative location of initial dividing surface. Default value 0.5.
            functional_kwargs (dict): Optional argiments for functionals. Pass feks.: functional_kwargs={"psi_disp": 1.5}

        Returns:
            (float): Grand potential
            (array): Grand potential contribution for each grid point
        """
        pif = PlanarInterface(thermopack=vle.eos,
                              temperature=vle.temperature,
                              domain_size=domain_size,
                              n_grid=n_grid,
                              functional_kwargs=functional_kwargs)
        pif.set_profile(vle,
                        profile,
                        invert_states=invert_states)
        return pif

    def surface_tension(self, reduced_unit=False):
        """
        Calculates the surface tension of the system.

        Args;
            reduced_unit (bool): Calculate using reduced units? Default False.

        Returns:
            (float): Surface tension (reduced units or J/m2)
        """

        if not self.converged:
            self.print_perform_minimization_message()
            return

        _, omega_a = self.grand_potential()
        omega_a += self.bulk.red_pressure_right * self.grid.integration_weights
        gamma = np.sum(omega_a)
        sigma = self.functional.thermo.sigma[0]
        gamma *= (sigma/self.functional.grid_reducing_lenght)**2
        if not reduced_unit:
            eps = self.functional.thermo.eps_div_kb[0] * KB
            gamma *= eps / sigma ** 2
        return gamma

    def parallel_pressure(self, reduced=True):
        """
        Calculates the parallel component of the pressure tensor

        Returns:
            (float): Reduced pressure
        """
        _, p_parallel = self.grand_potential()
        p_parallel = - p_parallel / self.grid.integration_weights
        return p_parallel*(self.functional.thermo.sigma[0]/self.functional.grid_reducing_lenght)**3

class SphericalInterface(Interface):
    """
    Utility class for simplifying specification of SPHERICAL interface
    """

    def __init__(self,
                 thermopack,
                 temperature,
                 radius,
                 domain_radius=100.0,
                 n_grid=1024,
                 specification = Specification.NUMBER_OF_MOLES,
                 functional_kwargs={}):
        """Class holding specifications for an interface calculation of a spherical geometry

        Args:
            thermopack (thermo): Thermopack instance
            temperature (float): Temperature (K)
            domain_size (float, optional): Sisze of domain. Defaults to 100.0.
            n_grid (int, optional): Number of grid points. Defaults to 1024.
            specification (Specification, optional): Override how system of equations are solved
            functional_kwargs (dict): Optional argiments for functionals. Pass feks.: functional_kwargs={"psi_disp": 1.5}
        Returns:
            None
        """
        Interface.__init__(self,
                           Geometry.SPHERICAL,
                           thermopack=thermopack,
                           temperature=temperature,
                           domain_size=domain_radius,
                           n_grid=n_grid,
                           specification=specification,
                           functional_kwargs=functional_kwargs)
        self.raduis = radius

    @staticmethod
    def from_tanh_profile(vle,
                          t_crit,
                          radius,
                          domain_radius=100.0,
                          n_grid=1024,
                          calculate_bubble=True,
                          sigma0=None,
                          specification=Specification.NUMBER_OF_MOLES,
                          functional_kwargs={}):
        """
        Initialize tangens hyperbolicus profile

            rel_pos_dividing_surface (float, optional): Relative location of initial dividing surface. Default value 0.5.
        Returns:
            (float): Grand potential
            (array): Grand potential contribution for each grid point
        """
        sif = SphericalInterface(thermopack=vle.eos,
                                 temperature=vle.temperature,
                                 radius=radius,
                                 domain_radius=domain_radius,
                                 n_grid=n_grid,
                                 specification=specification)
        if not sigma0:
            # Calculate planar surface tension
            sigma0 = sif.tanh_profile(vle,
                                      t_crit=t_crit,
                                      rel_pos_dividing_surface=0.5).solve().surface_tension()
        sif.sigma0 = sigma0
        # Extrapolate sigma 0
        phase = sif.functional.thermo.LIQPH if calculate_bubble else sif.functional.thermo.VAPPH
        real_radius = radius * sif.functional.grid_reducing_lenght
        signed_radius = real_radius * (-1.0 if calculate_bubble else 1.0)
        # Extrapolate chemical potential to first order and solve for phase densiteis
        mu, rho_l, rho_g = \
            sif.functional.thermo.extrapolate_mu_in_inverse_radius(sigma_0=sigma0,
                                                                    temp=vle.temperature,
                                                                    rho_l=vle.liquid.rho,
                                                                    rho_g=vle.vapor.rho,
                                                                    radius=signed_radius,
                                                                    geometry="SPHERICAL",
                                                                    phase=phase)
        #print(mu)
        #sys.exit()
        # Solve Laplace Extrapolate chemical potential to first order and solve for phase densiteis
        # mu, rho_l, rho_g = \
        #     sif.functional.thermo.solve_laplace(sigma_0=sigma0,
        #                                         temp=vle.temperature,
        #                                         rho_l=rho_l,
        #                                         rho_g=rho_g,
        #                                         radius=real_radius,
        #                                         geometry="SPHERICAL",
        #                                         phase=phase)

        rel_pos_dividing_surface = radius/domain_radius
        vapor = state(eos=vle.eos, T=vle.temperature, V=1/sum(rho_g), n=rho_g/sum(rho_g))
        liquid = state(eos=vle.eos, T=vle.temperature, V=1/sum(rho_l), n=rho_l/sum(rho_l))
        vle = equilibrium(vapor, liquid)
        # Set profile based on modefied densities
        sif.tanh_profile(vle=vle,
                         t_crit=t_crit,
                         rel_pos_dividing_surface=rel_pos_dividing_surface,
                         invert_states=not calculate_bubble)
        return sif

    @staticmethod
    def from_profile(vle,
                     profile,
                     domain_radius=100.0,
                     n_grid=1024,
                     invert_states=False,
                     specification=Specification.NUMBER_OF_MOLES,
                     functional_kwargs={}):
        """
        Initialize from profile

            functional_kwargs (dict): Optional argiments for functionals. Pass feks.: functional_kwargs={"psi_disp": 1.5}

        """
        sif = SphericalInterface(thermopack=vle.eos,
                                 temperature=vle.temperature,
                                 radius=0.0,
                                 domain_radius=domain_radius,
                                 n_grid=n_grid,
                                 specification=specification,
                                 functional_kwargs=functional_kwargs)

        sif.set_profile(vle,
                        profile,
                        invert_states=invert_states)
        return sif

    def surface_tension(self, reduced_unit=False):
        """
        Calculates the surface tension of the system.

        Args;
            reduced_unit (bool): Calculate using reduced units? Default False.

        Returns:
            (float): Surface tension (reduced units)
        """

        if not self.converged:
            self.print_perform_minimization_message()
            return

        _, omega_a = self.grand_potential()

        v_left = self.grid.get_volume(self.r_equimolar)
        v_right = self.grid.total_volume - v_left
        omega = np.sum(omega_a) + v_left*self.bulk.red_pressure_left + v_right*self.bulk.red_pressure_right
        gamma = omega / (4 * np.pi * self.r_equimolar**2)
        sigma = self.functional.thermo.sigma[0]
        gamma *= (sigma/self.functional.grid_reducing_lenght)**2
        if not reduced_unit:
            eps = self.functional.thermo.eps_div_kb[0] * KB
            gamma *= eps / sigma ** 2
        return gamma

    def work_of_formation(self):
        """
        Calculates the work of formation divided by kBT

        Returns:
            (float): Work of formation
        """

        if not self.converged:
            self.print_perform_minimization_message()
            return

        omega, _ = self.grand_potential()
        delta_omega = omega + self.grid.total_volume*self.bulk.red_pressure_right
        return delta_omega/self.bulk.reduced_temperature

    def n_excess(self):
        """
        Calculates the excess number of particles

        Returns:
            (float): Excess number of particles (mol)
        """

        if not self.profile:
            print("Need profile to calculate excess number of particles")
            return None

        n_excess = self.calculate_total_moles() - self.grid.total_volume*self.bulk.reduced_density_right
        return n_excess

    def surface_of_tension(self, reduced=False):
        """
        Calculates the surface tension of the system.

        Returns:
            (float): Surface tension (reduced or real units)
            (float): Surface of tension (reduced or real units)
            (float): Tolman length (reduced or real units)
        """

        if not self.converged:
            self.print_perform_minimization_message()
            return

        _, omega_a = self.grand_potential()
        delta_omega = np.sum(omega_a) + self.grid.total_volume*self.bulk.red_pressure_right
        dp = self.bulk.red_pressure_left - self.bulk.red_pressure_right
        gamma_s = (3*delta_omega*dp**2/16/np.pi)**(1/3)
        r_s = 2*gamma_s/dp
        delta = self.r_equimolar - r_s
        if reduced:
            sigma = self.functional.thermo.sigma[0]
            delta *= self.functional.grid_reducing_lenght/sigma
            r_s *= self.functional.grid_reducing_lenght/sigma
            gamma_s *= (sigma/self.functional.grid_reducing_lenght)**2
        else:
            eps = self.functional.thermo.eps_div_kb[0] * KB
            gamma_s *= eps / self.functional.grid_reducing_lenght ** 2
            r_s *= self.functional.grid_reducing_lenght
            delta *= self.functional.grid_reducing_lenght
        return gamma_s, r_s, delta

    def parallel_pressure(self, reduced=True):
        """
        Calculates the parallel component of the pressure tensor

        Returns:
            (float): Reduced pressure
        """
        print("Copied code form planar check validity")
        _, p_parallel = self.grand_potential()
        p_parallel = - p_parallel / self.grid.integration_weights
        return p_parallel*(self.functional.thermo.sigma[0]/self.functional.grid_reducing_lenght)**3

if __name__ == "__main__":
    pass
