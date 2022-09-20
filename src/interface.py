#!/usr/bin/env python3
import os
import sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from dft_numerics import dft_solver
from constants import NA, KB, Geometry, Specification, LenghtUnit, LCOLORS
from bulk import Bulk
from density_profile import Profile
from grid import Grid
from convolver import Convolver
from pyctp.pcsaft import pcsaft
from pyctp.saftvrmie import saftvrmie
from pyctp.saftvrqmie import saftvrqmie
from pyctp.ljs_bh import ljs_bh
from pyctp.ljs_wca import ljs_wca, ljs_uv
from pyctp.thermopack_state import state, equilibrium
from pcsaft_functional import pc_saft
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
        self.do_exp_mu = True
        self.convolver = None
        self.n_tot = None
        self.r_equimolar = None

    def unpack_profile(self, xvec):
        # Set profile
        n_grid = self.grid.n_grid
        n_c = self.functional.nc
        prof = Profile.empty_profile(n_c, n_grid)
        # Make sure boundary cells are set to bulk densities
        #self.mod_densities.assign_elements(self.densities)
        # Calculate weighted densities
        for ic in range(n_c):
            prof.densities[ic][:] = xvec[ic*n_grid:(ic+1)*n_grid]
        return prof

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
            #print("corr", self.convolver.correlation(0))
            integrals = self.integrate_df_vext()
            # #denum = np.dot(exp_beta_mu, integrals)
            #print("integrals",integrals)
            exp_mu = self.n_tot / integrals
            #print("exp_mu",exp_mu)
            #self.cDFT.mu_scaled_beta[:] = np.log(0.02276177)
            #exp_mu = np.exp(self.bulk.mu_scaled_beta)
            if self.do_exp_mu:
                xvec[n_rho:n_rho + n_c] = exp_mu
            else:
                xvec[n_rho:n_rho + n_c] = np.log(exp_mu)

        return xvec

    def residual(self, xvec):
        # Set profile
        n_grid = self.grid.n_grid
        n_c = self.functional.nc
        prof = self.unpack_profile(xvec)

        n_rho = n_c * n_grid
        beta_mu = np.zeros(n_c)
        if self.specification == Specification.NUMBER_OF_MOLES:
            if self.do_exp_mu:
                exp_beta_mu = np.zeros(n_c)
                exp_beta_mu[:] = xvec[n_rho:n_rho + n_c]
                beta_mu[:] = np.log(exp_beta_mu[:])
                #print(beta_mu[:])
            else:
                beta_mu[:] = xvec[n_rho:n_rho + n_c]
                exp_beta_mu = np.zeros(n_c)
                exp_beta_mu[:] = np.exp(beta_mu[:])
            exp_beta_mu_grid = np.zeros(n_c)
            integrals = self.integrate_df_vext()
            #print("integrals",integrals)
            denum = np.dot(exp_beta_mu, integrals)
            exp_beta_mu_grid[:] = self.n_tot * exp_beta_mu / denum
            beta_mu_grid = np.zeros(n_c)
            beta_mu_grid[:] = np.log(exp_beta_mu_grid[:])
            if self.grid.geometry is Geometry.PLANAR:
                # We know the chemical potential - use it
                beta_mu[:] = self.bulk.mu_scaled_beta[:]
        else:
            beta_mu[:] = self.bulk.mu_scaled_beta[:]

        # Perform convolution integrals
        self.convolver.convolve_density_profile(prof)

        # Calculate new density profile using the variations of the functional
        res = np.zeros(n_rho + n_c *
                       (1 if self.specification ==
                        Specification.NUMBER_OF_MOLES else 0))

        for ic in range(n_c):
            res[ic * n_grid:(ic+1)*n_grid] = - np.exp(self.convolver.correlation(ic)[:]
                       + beta_mu[:] - self.bulk.beta * self.v_ext[ic][:]) \
                + xvec[ic * n_grid:(ic+1)*n_grid]

        if self.specification == Specification.NUMBER_OF_MOLES:
            if self.do_exp_mu:
                res[n_rho:] = exp_beta_mu - exp_beta_mu_grid
            else:
                res[n_rho:] = beta_mu - beta_mu_grid

        return res

    def solve(self, solver=dft_solver(), log_iter=False):
        if not self.profile:
            self.converged = False
            print("Interface need to be initialized before calling solve")
        else:
            self.n_tot = self.calculate_total_mass()
            print("n_tot", self.n_tot)
            # Set up convolver
            self.convolver = Convolver(self.grid, self.functional, self.bulk.R, self.bulk.R_T)
            x0 = self.pack_x_vec()
            # print("x0:", x0)
            # sys.exit()
            x_sol, self.converged = solver.solve(
                x0, self.residual, log_iter)
            if self.converged:
                self.profile = self.unpack_profile(x_sol)
                # Update bulk properties
                rho_left = np.zeros_like(self.bulk.real_mu)
                rho_right = np.zeros_like(self.bulk.real_mu)
                for i in range(self.functional.nc):
                    rho_left[i] = self.profile.densities[i][1]
                    rho_right[i] = self.profile.densities[i][-2]
                self.bulk.update_bulk_densities(rho_left, rho_right)
                self.calculate_equimolar_dividing_surface()
            else:
                print("Interface solver did not converge")
        return self

    def single_convolution(self):
        # Set up convolver?
        if not self.convolver:
            self.convolver = Convolver(self.grid, self.functional, self.bulk.R, self.bulk.R_T)
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


    @abstractmethod
    def surface_tension(self):
        """
        Calculates the surface tension of the system.

        Returns:
            (float): Surface tension (reduced units)
        """

        if not self.converged:
            self.print_perform_minimization_message()
            return

        _, omega_a = self.grand_potential()

        print(self.bulk.red_pressure_left, self.bulk.red_pressure_right)

        v_left = self.grid.get_volume(self.r_equimolar)
        v_right = self.grid.total_volume - v_left
        print(v_left,v_right, sum(self.grid.integration_weights)-self.grid.total_volume)
        #gamma = np.sum(omega_a) + v_left*self.bulk.red_pressure_left + v_right*self.bulk.red_pressure_right

        delta_omega = np.sum(omega_a) + self.grid.total_volume*self.bulk.red_pressure_right
        dp = self.bulk.red_pressure_left - self.bulk.red_pressure_right
        gamma = (3*delta_omega*dp**2/16/np.pi)**(1/3)
        #omega_a += self.bulk.red_pressure_right * self.grid.integration_weights
        #gamma = np.sum(omega_a)
        return gamma

    def surface_tension_real_units(self):
        """
        Calculates the surface tension of the system.

        Returns:
            (float): Surface tension (J/m2)
        """
        gamma_star = self.surface_tension()
        eps = self.functional.thermo.eps_div_kb[0] * KB
        sigma = self.functional.grid_reducing_lenght
        gamma = gamma_star * eps / sigma ** 2

        return gamma


    def calculate_total_mass(self):
        """
        Calculates the overall mass of the system.

        Args:
            dens (densities): Density profiles

        Returns:
            (float): Surface tension (mol)
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

        N = self.calculate_total_mass()
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

    def get_excess_helmholtz_energy_density(self):
        """
        Calculates the Helmholtz energy in the system.

        Returns:
            (array): Helmholtz energy for each grid point
        """

        # FMT hard-sphere part
        F = self.bulk.reduced_temperature * self.functional.excess_free_energy(self.convolver.weighted_densities)
        return F

    def get_excess_entropy_density(self):
        """
        Get reduced entropy per reduced volume (-)
        """
        if not self.profile:
            print("Need profile to calculate entropy density")
            return

        eps = self.functional.thermo.eps_div_kb[0]
        f = self.functional.excess_free_energy(self.convolver.weighted_densities)
        f_T = self.convolver.functional_temperature_differential_convolution(self.profile.densities)
        vol_fac = (self.functional.thermo.sigma[0]/self.functional.grid_reducing_lenght)**3
        s = (- f - self.bulk.reduced_temperature * f_T * eps)*vol_fac
        return s

    def get_excess_entropy_density_real_units(self):
        """
        Get entropy per volume (J/m3/K)
        """
        s = self.get_excess_entropy_density()
        # Scale to real units
        s *= self.functional.thermo.Rgas/(NA*self.functional.thermo.sigma[0]**3)
        return s

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

    def plot_equilibrium_density_profiles(self,
                                          data_dict=None,
                                          xlim=None,
                                          ylim=None,
                                          plot_reduced_densities=False,
                                          plot_equimolar_surface=False,
                                          grid_unit=None):
        """
        Plot equilibrium density profile
        Args:
            data_dict: Additional data to plot
        """
        if not self.converged:
            self.print_perform_minimization_message()
            return
        if not grid_unit:
            grid_unit = self.functional.grid_unit
        fig, ax = plt.subplots(1, 1)
        if grid_unit==LenghtUnit.REDUCED:
            ax.set_xlabel("$z/\sigma_{11}$")
        elif grid_unit==LenghtUnit.ANGSTROM:
            ax.set_xlabel("$z$ (Å)")
        if self.functional.grid_unit == LenghtUnit.ANGSTROM and grid_unit == LenghtUnit.REDUCED:
            len_fac = 1.0/(self.functional.thermo.sigma[0]*1e10)
        else:
            len_fac = self.functional.grid_reducing_lenght*1e10
        dens_fac = np.ones(self.profile.densities.nc)
        if plot_reduced_densities:
            dens_fac *= (self.functional.thermo.sigma[0]/self.functional.grid_reducing_lenght)**3
            ax.set_ylabel(r"$\rho^*$")
        else:
            dens_fac *= 1.0e-3/(NA*self.functional.grid_reducing_lenght**3)
            ax.set_ylabel(r"$\rho$ (kmol/m$^3$)")
        for i in range(self.profile.densities.nc):
            ax.plot(self.grid.z[:]*len_fac,
                    self.profile.densities[i][:] * dens_fac[i],
                    lw=2, color=LCOLORS[i], label=f"Comp. {i+1}")
        if data_dict is not None:
            plot_data_container(data_dict, ax)

        if xlim is not None:
            plt.xlim(xlim)
        if ylim is not None:
            plt.ylim(ylim)

        if plot_equimolar_surface:
            # Plot equimolar dividing surface
            yl = ax.get_ylim()
            ax.plot([len_fac*self.r_equimolar, len_fac*self.r_equimolar],
                    [0.0, yl[1]],
                    lw=1, color="k",
                    linestyle="--",
                    label="Eq. mol. surf.")

        leg = plt.legend(loc="best", numpoints=1, frameon=False)

        filename = self.generate_case_name() + ".pdf"
        plt.savefig(filename)
        plt.show()

    def grand_potential_bulk(self, wdens, Vext=0.0):
        """
        Calculates the grand potential in the system in bulk.
        Method used for testing.
        Args:
            dens : Weigthed densities
            Vext (float): External potential in bulk

        Returns:
            (float): Grand potential per volume
        """

        # Calculate chemical potential (excess + ideal)
        mu = self.T * (self.mu_res_scaled_beta + np.log(self.bulk_densities))

        # FMT hard-sphere part
        omega_a = self.T * \
            self.functional.excess_free_energy(wdens)

        # Add ideal part and extrinsic part
        for i in range(self.nc):
            # Ideal part
            omega_a[:] += self.T * self.bulk_densities[i] * \
                (np.log(self.bulk_densities[i]) - 1.0)
            # Extrinsic part
            omega_a[:] += self.bulk_densities[i] \
                * (Vext - mu[i])

        return omega_a[0]

    def test_grand_potential_bulk(self):
        """
        """
        # Test grand potential in bulk phase
        wdens = weighted_densities_1D(
            1, self.functional.R, ms=np.ones(self.nc))
        wdens.set_testing_values(rho=self.bulk_densities)
        wdens.n1v[:] = 0.0
        wdens.n2v[:] = 0.0
        omega = self.grand_potential_bulk(wdens, Vext=0.0)
        print("omega:", omega)
        print("pressure + omega:", self.red_pressure + omega)

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

    def surface_tension(self):
        """
        Calculates the surface tension of the system.

        Returns:
            (float): Surface tension (reduced units)
        """

        if not self.converged:
            self.print_perform_minimization_message()
            return

        _, omega_a = self.grand_potential()
        omega_a += self.bulk.red_pressure_right * self.grid.integration_weights
        gamma = np.sum(omega_a)
        return gamma

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
                          sigma0=None):
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
                                 n_grid=n_grid)
        if not sigma0:
            # Calculate planar surface tension
            sigma0 = sif.tanh_profile(vle,
                                      t_crit=t_crit,
                                      rel_pos_dividing_surface=0.5).solve().surface_tension_real_units()
        sif.sigma0 = sigma0
        print(sigma0)
        # Extrapolate sigma 0
        phase = sif.functional.thermo.LIQPH if calculate_bubble else sif.functional.thermo.VAPPH
        print(phase)
        real_radius = radius * sif.functional.grid_reducing_lenght
        print(real_radius)
        print(vle.vapor.rho,vle.liquid.rho)
        # Extrapolate chemical potential to first order and solve for phase densiteis
        mu, rho_l, rho_g = \
            sif.functional.thermo.extrapolate_mu_in_inverse_radius(sigma_0=sigma0,
                                                                    temp=vle.temperature,
                                                                    rho_l=vle.liquid.rho,
                                                                    rho_g=vle.vapor.rho,
                                                                    radius=real_radius,
                                                                    geometry="SPHERICAL",
                                                                    phase=phase)
        # Solve Laplace Extrapolate chemical potential to first order and solve for phase densiteis
        mu, rho_l, rho_g = \
            sif.functional.thermo.solve_laplace(sigma_0=sigma0,
                                                temp=vle.temperature,
                                                rho_l=rho_l,
                                                rho_g=rho_g,
                                                radius=real_radius,
                                                geometry="SPHERICAL",
                                                phase=phase)

        rel_pos_dividing_surface = radius/domain_radius
        vapor = state(eos=vle.eos, T=vle.temperature, V=1/sum(rho_g), n=rho_g/sum(rho_g))
        liquid = state(eos=vle.eos, T=vle.temperature, V=1/sum(rho_l), n=rho_l/sum(rho_l))
        vle_modified = equilibrium(vapor, liquid)
        # Set profile based on modefied densities
        sif.tanh_profile(vle=vle_modified,
                         t_crit=t_crit,
                         rel_pos_dividing_surface=rel_pos_dividing_surface,
                         invert_states=not calculate_bubble)
        return sif

    def surface_tension(self):
        """
        Calculates the surface tension of the system.

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
        return gamma

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
        if not reduced:
            eps = self.functional.thermo.eps_div_kb[0] * KB
            sigma = self.functional.grid_reducing_lenght
            gamma_s *= eps / sigma ** 2
            r_s *= sigma
            delta *= sigma
        return gamma_s, r_s, delta

if __name__ == "__main__":
    pass
